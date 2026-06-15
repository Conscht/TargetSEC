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
SYSTEMS = {
    "TargetSEC": (
        "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
        "eval_outputs/finetune_epoch48_guidance4_gs07/wav"
    ),
    "HiFiGAN": (
        "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
        "eval_outputs/hifigan_baseline_epoch116/wav"
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000


def load_wav(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    return wav.squeeze(0)  # (T,)


def pad_batch(wavs: list) -> torch.Tensor:
    """Pad list of 1-D tensors to same length → (B, T)."""
    max_len = max(w.size(0) for w in wavs)
    return torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.size(0))) for w in wavs
    ])


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

        batch = pad_batch(wavs).to(DEVICE)   # (B, T) where B ≤ 7
        with torch.no_grad():
            out_prob, _, _, _ = classifier.classify_batch(batch)

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
