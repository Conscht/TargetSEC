"""
ECAPA-TDNN speaker similarity: Converted vs Ground Truth (GT)

Assumes your converted audio is saved as:
  <converted_root>/class_{1..7}/utt_{i:06d}.wav

And GT audio for index i corresponds to the i-th item in the sorted mel list:
  <test_tensor_dir>/*.pt  (sorted)
  GT wav path = <gt_audio_dir>/<mel_name.replace("_mel.pt",".wav")>

Outputs:
- Per-class mean/std/median/q25/q75/min/max
- Overall pooled statistics
- Prints a table
- Saves JSON + CSV
"""

import os
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import logging
from speechbrain.pretrained import SpeakerRecognition

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# ECAPA wrapper
# -----------------------------
class ECAPA:
    def __init__(self, device=device, model_hub="speechbrain/spkrec-ecapa-voxceleb"):
        self.device = device
        logger.info(f"Loading ECAPA model: {model_hub}")
        self.model = SpeakerRecognition.from_hparams(
            source=model_hub,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)},
        )

    @torch.inference_mode()
    def embed(self, wav_path: str) -> torch.Tensor | None:
        try:
            wav, sr = torchaudio.load(wav_path)  # (C,T)

            # mono
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # resample to 16k
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)

            # (batch, time)
            wav = wav.to(self.device)  # (1,T)
            emb = self.model.encode_batch(wav).squeeze()

            # ensure (D,)
            if emb.dim() != 1:
                emb = emb.view(-1)

            return emb.detach().cpu()
        except Exception as e:
            logger.warning(f"Embedding failed for {wav_path}: {e}")
            return None

    def cosine(self, e1: torch.Tensor, e2: torch.Tensor) -> float | None:
        if e1 is None or e2 is None:
            return None
        return torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()


# -----------------------------
# Helpers
# -----------------------------
def get_sorted_mel_files(test_tensor_dir: str) -> list[str]:
    files = [f for f in os.listdir(test_tensor_dir) if f.endswith("_mel.pt")]
    files.sort()
    return files


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {
            "num_scored": 0,
            "mean": None, "std": None, "median": None,
            "q25": None, "q75": None, "min": None, "max": None
        }
    return {
        "num_scored": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def format_table(rows, headers):
    """
    Minimal pretty table (no extra deps).
    rows: list[list[str]]
    headers: list[str]
    """
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(r):
        return " | ".join(str(v).ljust(w) for v, w in zip(r, widths))

    sep = "-+-".join("-" * w for w in widths)

    out = []
    out.append(fmt_row(headers))
    out.append(sep)
    for r in rows:
        out.append(fmt_row(r))
    return "\n".join(out)


def save_csv(per_class_stats: dict, overall_stats: dict, out_csv: str):
    """
    Writes a simple CSV with per-class rows + overall row.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    cols = ["split", "num_scored", "mean", "std", "median", "q25", "q75", "min", "max"]
    lines = [",".join(cols)]

    for c in sorted(per_class_stats.keys(), key=int):
        st = per_class_stats[c]
        row = [
            f"class_{c}",
            str(st["num_scored"]),
            "" if st["mean"] is None else f"{st['mean']:.6f}",
            "" if st["std"] is None else f"{st['std']:.6f}",
            "" if st["median"] is None else f"{st['median']:.6f}",
            "" if st["q25"] is None else f"{st['q25']:.6f}",
            "" if st["q75"] is None else f"{st['q75']:.6f}",
            "" if st["min"] is None else f"{st['min']:.6f}",
            "" if st["max"] is None else f"{st['max']:.6f}",
        ]
        lines.append(",".join(row))

    st = overall_stats
    lines.append(",".join([
        "overall",
        str(st["num_scored"]),
        "" if st["mean"] is None else f"{st['mean']:.6f}",
        "" if st["std"] is None else f"{st['std']:.6f}",
        "" if st["median"] is None else f"{st['median']:.6f}",
        "" if st["q25"] is None else f"{st['q25']:.6f}",
        "" if st["q75"] is None else f"{st['q75']:.6f}",
        "" if st["min"] is None else f"{st['min']:.6f}",
        "" if st["max"] is None else f"{st['max']:.6f}",
    ]))

    with open(out_csv, "w") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main evaluation
# -----------------------------
def eval_gt_vs_converted_table(
    test_tensor_dir: str,
    gt_audio_dir: str,
    converted_root: str,
    out_json: str | None = None,
    out_csv: str | None = None,
):
    model = ECAPA(device=device)

    mel_files = get_sorted_mel_files(test_tensor_dir)
    logger.info(f"Found {len(mel_files)} test items in mel dir (sorted).")

    gt_audio_dir = Path(gt_audio_dir)
    converted_root = Path(converted_root)

    # cache GT embeddings by index
    gt_emb_cache: dict[int, torch.Tensor] = {}

    per_class = {}
    pooled = []

    for c in range(1, 8):
        class_dir = converted_root / f"class_{c}"
        if not class_dir.exists():
            logger.warning(f"Missing converted dir: {class_dir}")
            continue

        sims = []
        missing_pairs = 0

        for i, mel_name in enumerate(mel_files):
            if i % 200 == 0 and i > 0:
                logger.info(f"class_{c}: processed {i}/{len(mel_files)}")

            gt_name = mel_name.replace("_mel.pt", ".wav")
            gt_path = gt_audio_dir / gt_name
            conv_path = class_dir / f"utt_{i:06d}.wav"

            if (not gt_path.exists()) or (not conv_path.exists()):
                missing_pairs += 1
                continue

            if i not in gt_emb_cache:
                gt_emb_cache[i] = model.embed(str(gt_path))

            e_gt = gt_emb_cache[i]
            e_conv = model.embed(str(conv_path))
            sim = model.cosine(e_gt, e_conv)
            if sim is not None:
                sims.append(sim)

        sims = np.array(sims, dtype=np.float32) if sims else np.array([], dtype=np.float32)
        stats = summarize(sims)
        stats["missing_pairs"] = int(missing_pairs)
        stats["num_items_expected"] = int(len(mel_files))
        per_class[c] = stats

        pooled.extend(sims.tolist())
        logger.info(f"class_{c}: scored={stats['num_scored']} missing={missing_pairs} mean={stats['mean']}")

    pooled = np.array(pooled, dtype=np.float32) if pooled else np.array([], dtype=np.float32)
    overall = summarize(pooled)

    results = {"per_class": per_class, "overall": overall}

    # Print table
    headers = ["Class", "N", "Mean", "Std", "Median", "Q25", "Q75", "Min", "Max", "Missing"]
    rows = []
    for c in sorted(per_class.keys()):
        st = per_class[c]
        rows.append([
            str(c),
            str(st["num_scored"]),
            "-" if st["mean"] is None else f"{st['mean']:.4f}",
            "-" if st["std"] is None else f"{st['std']:.4f}",
            "-" if st["median"] is None else f"{st['median']:.4f}",
            "-" if st["q25"] is None else f"{st['q25']:.4f}",
            "-" if st["q75"] is None else f"{st['q75']:.4f}",
            "-" if st["min"] is None else f"{st['min']:.4f}",
            "-" if st["max"] is None else f"{st['max']:.4f}",
            str(st.get("missing_pairs", 0)),
        ])

    # overall row
    rows.append([
        "OVERALL",
        str(overall["num_scored"]),
        "-" if overall["mean"] is None else f"{overall['mean']:.4f}",
        "-" if overall["std"] is None else f"{overall['std']:.4f}",
        "-" if overall["median"] is None else f"{overall['median']:.4f}",
        "-" if overall["q25"] is None else f"{overall['q25']:.4f}",
        "-" if overall["q75"] is None else f"{overall['q75']:.4f}",
        "-" if overall["min"] is None else f"{overall['min']:.4f}",
        "-" if overall["max"] is None else f"{overall['max']:.4f}",
        "-",
    ])

    print("\nECAPA Speaker Similarity: Converted vs Ground Truth")
    print(format_table(rows, headers))
    print()

    # Save
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON: {out_json}")

    if out_csv:
        save_csv(per_class, overall, out_csv)
        logger.info(f"Saved CSV:  {out_csv}")

    return results


if __name__ == "__main__":
    # Set these to your paths
    TEST_TENSOR_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Test1"
    GT_AUDIO_DIR    = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/Audio"
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--wav_root", default="/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/eval_outputs/ldm_finetune_epoch596/wav",
                    help="<root>/class_{1..7}/utt_%%06d.wav for the system under test")
    _a, _ = _p.parse_known_args()
    CONVERTED_ROOT  = _a.wav_root

    OUT_JSON = os.path.join(os.path.dirname(CONVERTED_ROOT.rstrip("/")), "ecapa_gt_vs_converted.json")
    OUT_CSV  = os.path.join(os.path.dirname(CONVERTED_ROOT.rstrip("/")), "ecapa_gt_vs_converted.csv")

    eval_gt_vs_converted_table(
        test_tensor_dir=TEST_TENSOR_DIR,
        gt_audio_dir=GT_AUDIO_DIR,
        converted_root=CONVERTED_ROOT,
        out_json=OUT_JSON,
        out_csv=OUT_CSV,
    )
