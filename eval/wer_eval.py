"""
WER evaluation for intelligibility.

Runs Whisper-medium on:
  - Ground truth audio (as ASR reference)
  - TargetSEC class_4 WAVs (hypothesis)

class_4 is used as a representative mid-arousal condition — WER should be
class-invariant since emotion conversion preserves linguistic content.

Audio is peak-normalized before transcription: Whisper does not normalize
loudness internally (unlike WVMOS, which z-score normalizes via
Wav2Vec2Processor), so quiet low-arousal audio could otherwise be
disadvantaged relative to louder high-arousal audio.

Reports corpus-level WER (aggregate edit distance / aggregate reference
word count via jiwer.process_words) — the standard ASR convention — plus
mean/median of per-utterance ratios for reference. Corpus-level WER is
far less sensitive to outlier utterances than the mean of per-utterance
ratios.

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
TARGETSEC_WAV_DIR = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "eval_outputs/ldm_finetune_epoch596/wav/class_4"
)

N_SAMPLES = None   # full test set (16,903 utterances)
WHISPER_MODEL = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PEAK_TARGET = 0.95  # normalize peak amplitude to this before transcription

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

    entries = read_manifest(TEST1_MANIFEST)
    total = len(entries) if N_SAMPLES is None else min(N_SAMPLES, len(entries))
    print(f"Evaluating {total} utterances (TargetSEC only, peak-normalized)...")

    refs, hyps = [], []
    per_utt_wers = []
    skipped = 0

    for i, basename in enumerate(entries[:total]):
        gt_path = os.path.join(GT_AUDIO_DIR, basename)
        ts_path = os.path.join(TARGETSEC_WAV_DIR, f"utt_{i:06d}.wav")

        if not os.path.exists(gt_path):
            skipped += 1
            continue

        gt_audio = load_audio_as_np(gt_path)
        ref_text = model.transcribe(gt_audio, language="en")["text"].strip()
        if not ref_text:
            continue

        if os.path.exists(ts_path):
            ts_audio = load_audio_as_np(ts_path)
            hyp = model.transcribe(ts_audio, language="en")["text"].strip()
            refs.append(ref_text)
            hyps.append(hyp)
            try:
                wer = jiwer.wer(ref_text, hyp, reference_transform=TRANSFORM,
                                hypothesis_transform=TRANSFORM)
                per_utt_wers.append(wer)
            except Exception:
                pass
        else:
            skipped += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] n={len(refs)} collected, {skipped} skipped")

    print("\n=== WER Results (Whisper-medium, class_4, full test set, peak-normalized) ===")
    if refs:
        corpus_wer = jiwer.wer(refs, hyps, reference_transform=TRANSFORM,
                                hypothesis_transform=TRANSFORM)
        print(f"  Corpus-level WER : {corpus_wer*100:.1f}%  (N={len(refs)} utterances, standard ASR convention)")
        print(f"  Mean per-utt WER : {np.mean(per_utt_wers)*100:.1f}% ± {np.std(per_utt_wers)*100:.1f}%")
        print(f"  Median per-utt   : {np.median(per_utt_wers)*100:.1f}%")
    if skipped:
        print(f"  [skipped {skipped} utterances — GT or TargetSEC WAV not found]")


if __name__ == "__main__":
    main()
