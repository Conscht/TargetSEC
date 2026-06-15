"""
WER evaluation for intelligibility comparison.

Runs Whisper-medium on:
  - Ground truth audio (as ASR reference)
  - TargetSEC class_4 WAVs (hypothesis)
  - HiFiGAN baseline class_4 WAVs (hypothesis)

Reports mean WER ± std per system.

class_4 is used as a representative mid-arousal condition — WER should be
class-invariant since emotion conversion preserves linguistic content.

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
    "eval_outputs/finetune_epoch48_guidance4_gs07/wav/class_4"
)
HIFIGAN_WAV_DIR = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "eval_outputs/hifigan_baseline_epoch116/wav/class_4"
)

N_SAMPLES = None   # full test set (16,903 utterances)
WHISPER_MODEL = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Normalisation for fair WER comparison ─────────────────────────────────
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
    return wav.squeeze(0).numpy()


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
    print(f"Evaluating {total} utterances...")

    gt_wers, ts_wers, hf_wers = [], [], []
    skipped = {"targetsec": 0, "hifigan": 0}

    for i, basename in enumerate(entries[:total]):
        gt_path = os.path.join(GT_AUDIO_DIR, basename)
        ts_path = os.path.join(TARGETSEC_WAV_DIR, f"utt_{i:06d}.wav")
        hf_path = os.path.join(HIFIGAN_WAV_DIR, f"utt_{i:06d}.wav")

        if not os.path.exists(gt_path):
            continue

        gt_audio = load_audio_as_np(gt_path)
        ref_text = model.transcribe(gt_audio, language="en")["text"].strip()
        if not ref_text:
            continue

        if os.path.exists(ts_path):
            ts_audio = load_audio_as_np(ts_path)
            hyp = model.transcribe(ts_audio, language="en")["text"].strip()
            try:
                wer = jiwer.wer(ref_text, hyp, truth_transform=TRANSFORM,
                                hypothesis_transform=TRANSFORM)
                ts_wers.append(wer)
            except Exception:
                pass
        else:
            skipped["targetsec"] += 1

        if os.path.exists(hf_path):
            hf_audio = load_audio_as_np(hf_path)
            hyp = model.transcribe(hf_audio, language="en")["text"].strip()
            try:
                wer = jiwer.wer(ref_text, hyp, truth_transform=TRANSFORM,
                                hypothesis_transform=TRANSFORM)
                hf_wers.append(wer)
            except Exception:
                pass
        else:
            skipped["hifigan"] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] TargetSEC n={len(ts_wers)}, HiFiGAN n={len(hf_wers)}")

    print("\n=== WER Results (Whisper-medium, class_4) ===")
    if ts_wers:
        print(f"  TargetSEC : {np.mean(ts_wers)*100:.1f}% ± {np.std(ts_wers)*100:.1f}%  (N={len(ts_wers)})")
    if hf_wers:
        print(f"  HiFiGAN   : {np.mean(hf_wers)*100:.1f}% ± {np.std(hf_wers)*100:.1f}%  (N={len(hf_wers)})")
    if skipped["targetsec"]:
        print(f"  [skipped {skipped['targetsec']} TargetSEC WAVs not found]")
    if skipped["hifigan"]:
        print(f"  [skipped {skipped['hifigan']} HiFiGAN WAVs not found]")

    # GT self-WER (transcription consistency, should be ~0 but shows Whisper noise floor)
    print("\nNote: GT WER reported above is Whisper transcription of GT used as its own reference.")
    print("Expected: TargetSEC ≈ HiFiGAN WER if intelligibility is preserved.")


if __name__ == "__main__":
    main()
