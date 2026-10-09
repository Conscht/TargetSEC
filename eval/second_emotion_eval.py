"""
Independent emotion evaluation using SpeechBrain IEMOCAP model.

Addresses reviewer concern about circular evaluation: training uses
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (dimensional, MSP-Podcast).
This script uses speechbrain/emotion-recognition-wav2vec2-IEMOCAP (categorical,
IEMOCAP) — different architecture, different training data.

Arousal proxy: P(angry) + P(happy)  [high-activation IEMOCAP categories]

Metric: Spearman ρ between target class (1–7) and proxy arousal score,
computed over all 16,903 test utterances × 7 classes = 118,321 samples per system.

Batches all 7 class WAVs per utterance together for efficient GPU inference.
"""
import ast
import os
import types
import numpy as np
import torch
import torchaudio
from scipy.stats import spearmanr
from speechbrain.inference.classifiers import EncoderClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
TEST1_MANIFEST = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"
)
import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--wav_root", action="append", default=None,
                help="Repeatable: <root>/class_{1..7}. Defaults to TargetSEC.")
_p.add_argument("--name", action="append", default=None, help="Label per --wav_root")
_a, _ = _p.parse_known_args()
if _a.wav_root:
    _names = _a.name or [f"sys{i}" for i in range(len(_a.wav_root))]
    SYSTEMS = dict(zip(_names, _a.wav_root))
else:
    SYSTEMS = {"TargetSEC": ("/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
                             "Code/EmoConv-LDM/eval_outputs/ldm_finetune_epoch596/wav")}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000


def load_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    return wav.squeeze(0)  # (T,)


def pad_batch(wavs: list) -> tuple:
    """Pad 1-D tensors → (B, T) and return relative lengths."""
    lengths = [w.size(0) for w in wavs]
    max_len = max(lengths)
    padded = torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.size(0))) for w in wavs
    ])
    wav_lens = torch.tensor([l / max_len for l in lengths], dtype=torch.float32)
    return padded, wav_lens


def _patched_encode_batch(self, wavs, wav_lens=None, normalize=False):
    """encode_batch patched for wav2vec2-based models (no compute_features module).

    Mods available: wav2vec2, avg_pool, output_mlp — no compute_features or classifier.
    """
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)
    if wav_lens is None:
        wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
    wavs = wavs.to(self.device)
    wav_lens = wav_lens.to(self.device)
    feats = self.mods.wav2vec2(wavs, wav_lens)
    emb = self.mods.avg_pool(feats, wav_lens)
    return emb


def _patched_classify_batch(self, wavs, wav_lens=None):
    """classify_batch patched: uses output_mlp instead of non-existent mods.classifier."""
    emb = self.encode_batch(wavs, wav_lens)
    out_logits = self.mods.output_mlp(emb).squeeze(1)  # (B, n_classes)
    out_prob = torch.nn.functional.log_softmax(out_logits, dim=-1)
    score, index = torch.max(out_prob, dim=-1)
    text_lab = self.hparams.label_encoder.decode_torch(index)
    return out_prob, score, index, text_lab


def build_label_map(classifier) -> dict:
    return {v: k for k, v in classifier.hparams.label_encoder.ind2lab.items()}


def eval_system(classifier, label_map, wav_dir, n_entries):
    ang_idx = label_map.get("ang", None)
    hap_idx = label_map.get("hap", None)

    target_classes, proxy_scores = [], []
    skipped = 0

    for i in range(n_entries):
        wavs = []
        valid_classes = []
        for c in range(1, 8):
            p = os.path.join(wav_dir, f"class_{c}", f"utt_{i:06d}.wav")
            if not os.path.exists(p):
                skipped += 1
                continue
            try:
                wavs.append(load_wav(p))
                valid_classes.append(c)
            except Exception:
                skipped += 1

        if not wavs:
            continue

        batch, wav_lens = pad_batch(wavs)
        batch = batch.to(DEVICE)
        wav_lens = wav_lens.to(DEVICE)

        with torch.no_grad():
            out_prob, _, _, _ = classifier.classify_batch(batch, wav_lens)

        probs = out_prob.exp()  # log-softmax → softmax, shape (B, n_classes)
        for j, c in enumerate(valid_classes):
            ang_p = probs[j, ang_idx].item() if ang_idx is not None else 0.0
            hap_p = probs[j, hap_idx].item() if hap_idx is not None else 0.0
            target_classes.append(c)
            proxy_scores.append(ang_p + hap_p)

        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{n_entries}] {len(proxy_scores)} scores, {skipped} skipped")

    return np.array(target_classes), np.array(proxy_scores)


def main():
    print(f"Loading SpeechBrain IEMOCAP classifier on {DEVICE}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="pretrained_models/emotion-recognition-iemocap",
        run_opts={"device": str(DEVICE)},
    )
    classifier.eval()

    # Patch for wav2vec2 model: mods has wav2vec2/avg_pool/output_mlp, not compute_features/classifier
    print(f"Available mods: {list(classifier.mods.keys())}")
    classifier.encode_batch = types.MethodType(_patched_encode_batch, classifier)
    classifier.classify_batch = types.MethodType(_patched_classify_batch, classifier)

    label_map = build_label_map(classifier)
    print(f"IEMOCAP label map: {label_map}")
    if "ang" not in label_map or "hap" not in label_map:
        raise RuntimeError(f"Expected 'ang'/'hap' labels, got: {list(label_map.keys())}")

    with open(TEST1_MANIFEST) as f:
        entries = f.readlines()
    n = len(entries)
    print(f"Test set: {n} utterances × 7 classes = {n*7} samples per system\n")

    results = {}
    for name, wav_dir in SYSTEMS.items():
        print(f"--- {name} ---")
        classes, scores = eval_system(classifier, label_map, wav_dir, n)
        results[name] = (classes, scores)
        print(f"    Done: {len(scores)} scores collected\n")

    print("\n=== Independent Arousal Evaluation (SpeechBrain / IEMOCAP) ===")
    print("Arousal proxy = P(angry) + P(happy)\n")

    for name, (classes, scores) in results.items():
        if len(scores) == 0:
            print(f"  {name}: no data\n")
            continue
        rho, pval = spearmanr(classes, scores)
        print(f"  {name}: Spearman ρ = {rho:.3f}  p = {pval:.2e}  N = {len(scores)}")
        for c in range(1, 8):
            mask = classes == c
            if mask.any():
                print(f"    Class {c}: proxy = {scores[mask].mean():.3f} ± {scores[mask].std():.3f}")
        print()


if __name__ == "__main__":
    main()
