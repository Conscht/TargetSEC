"""
WER per-class verification: 200 utterances × 7 arousal classes = 1,400 samples.

Verifies that WER is class-invariant (content preservation is independent of
target arousal). Audio is peak-normalized before transcription (Whisper does
not normalize loudness internally), so quiet low-arousal audio isn't
disadvantaged relative to louder high-arousal audio.

Reports corpus-level WER per class (aggregate edit distance / aggregate
reference word count) rather than the mean of per-utterance ratios.

Requires: pip install openai-whisper jiwer
"""
import ast
import os
import numpy as np
import torch
import torchaudio
import whisper
import jiwer

# ── Paths ──────────────────────────────────────────────────────────────────
TEST1_MANIFEST = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"
)
GT_AUDIO_DIR = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/Audio"
)
import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--sample_n", type=int, default=0,
                help="Random sample of N utterances (0 = legacy first-N). Use the "
                     "same seed as the WVMOS/arousal probes so every metric lands "
                     "on one subset and paired per-utterance tests are possible.")
_p.add_argument("--sample_seed", type=int, default=0)
_p.add_argument("--wav_root", default="/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/eval_outputs/ldm_finetune_epoch596/wav")
_a, _ = _p.parse_known_args()
TARGETSEC_WAV_ROOT = _a.wav_root

N_PER_CLASS = _a.sample_n if _a.sample_n else 200
WHISPER_MODEL = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PEAK_TARGET = 0.95

TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])


def load_audio_as_np(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    audio = wav.squeeze(0).numpy()
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak * PEAK_TARGET
    return audio



def _words(txt):
    out = TRANSFORM(txt)
    return out[0] if out and isinstance(out[0], list) else out


def corpus_wer(refs, hyps):
    """sum(edits)/sum(reference words), robust to empty hypotheses.

    jiwer.wer(list, list) raises when RemoveEmptyStrings drops an empty
    hypothesis but keeps its reference -- the crash that killed
    wer_eval-2084357 and the first 2000-utt run here. Dropping those pairs
    would also bias WER downward: an empty transcription is a total failure,
    so it is scored as deletion of every reference word.
    """
    tot_err = tot_ref = n_empty = 0
    for r, h in zip(refs, hyps):
        rw = _words(r)
        if not rw:
            continue
        hw = _words(h) if h.strip() else []
        if not hw:
            n_empty += 1
            tot_err += len(rw); tot_ref += len(rw)
            continue
        m = jiwer.process_words(" ".join(rw), " ".join(hw))
        tot_err += m.substitutions + m.deletions + m.insertions
        tot_ref += m.substitutions + m.deletions + m.hits
    return (tot_err / tot_ref if tot_ref else float("nan")), n_empty


def read_manifest(path: str):
    entries = []
    with open(path) as f:
        for line in f:
            data = ast.literal_eval(line.strip())
            basename = os.path.basename(data["audio"])
            entries.append(basename)
    return entries


def main():
    print(f"Loading Whisper {WHISPER_MODEL} on {DEVICE}...")
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

    # Index by the SAME order the benchmark wrote wavs in (sorted *_mel.pt),
    # not the manifest: they agree at 16829/16903 positions but diverge at 74
    # (indices 10121-12387) because of names like MSP-PODCAST_1214_0008.wav vs
    # MSP-PODCAST_1214_0008_0001.wav. Manifest indexing pairs those with the
    # wrong reference.
    _MEL_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Test1"
    entries = [f.replace("_mel.pt", ".wav")
               for f in sorted(x for x in os.listdir(_MEL_DIR) if x.endswith("_mel.pt"))]
    assert set(entries) == set(read_manifest(TEST1_MANIFEST))
    print(f"Manifest: {len(entries)} utterances. Sampling {N_PER_CLASS} per class (peak-normalized).\n")

    gt_cache = {}   # i → ref_text
    needed = N_PER_CLASS
    collected = 0
    _order = list(enumerate(entries))
    if _a.sample_n:
        import random as _rnd
        _rnd.Random(_a.sample_seed).shuffle(_order)
        print(f"Random sample: {needed} of {len(entries)} (seed {_a.sample_seed})", flush=True)
    for i, basename in _order:
        if collected >= needed:
            break
        gt_path = os.path.join(GT_AUDIO_DIR, basename)
        if not os.path.exists(gt_path):
            continue
        gt_audio = load_audio_as_np(gt_path)
        ref_text = model.transcribe(gt_audio, language="en")["text"].strip()
        if ref_text:
            gt_cache[i] = ref_text
            collected += 1
    print(f"Cached {len(gt_cache)} GT transcriptions.\n")

    per_class_refs = {}
    per_class_hyps = {}
    per_class_per_utt = {}

    for c in range(1, 8):
        class_dir = os.path.join(TARGETSEC_WAV_ROOT, f"class_{c}")
        refs, hyps, wers = [], [], []

        for i, ref_text in gt_cache.items():
            ts_path = os.path.join(class_dir, f"utt_{i:06d}.wav")
            if not os.path.exists(ts_path):
                continue
            ts_audio = load_audio_as_np(ts_path)
            hyp = model.transcribe(ts_audio, language="en")["text"].strip()
            refs.append(ref_text)
            hyps.append(hyp)
            try:
                wer = jiwer.wer(ref_text, hyp, reference_transform=TRANSFORM,
                                hypothesis_transform=TRANSFORM)
                wers.append(wer)
            except Exception:
                pass

        per_class_refs[c] = refs
        per_class_hyps[c] = hyps
        per_class_per_utt[c] = np.array(wers, dtype=np.float32)

        corpus_wer_val, n_empty = corpus_wer(refs, hyps) if refs else (float("nan"), 0)
        print(f"  class_{c}: corpus WER = {corpus_wer_val*100:.1f}%  "
              f"(mean per-utt = {np.mean(wers)*100:.1f}% ± {np.std(wers)*100:.1f}%, N={len(refs)})")

    all_refs = sum(per_class_refs.values(), [])
    all_hyps = sum(per_class_hyps.values(), [])
    all_corpus_wer, all_empty = corpus_wer(all_refs, all_hyps)

    print(f"\n=== WER Per-Class Summary (TargetSEC, {N_PER_CLASS} utterances/class, peak-normalized) ===")
    print(f"{'Class':<8} {'Corpus WER':>11} {'Mean':>8} {'Median':>8} {'N':>6}")
    print("-" * 48)
    for c in range(1, 8):
        w = per_class_per_utt[c]
        corpus_wer_val, _ = corpus_wer(per_class_refs[c], per_class_hyps[c])
        if w.size:
            print(f"  {c:<6} {corpus_wer_val*100:>10.1f}% {np.mean(w)*100:>7.1f}% {np.median(w)*100:>7.1f}% {w.size:>6}")
    print("-" * 48)
    print(f"  {'ALL':<6} {all_corpus_wer*100:>10.1f}%  {'':>7} {'':>7} {len(all_refs):>6}")


if __name__ == "__main__":
    main()
