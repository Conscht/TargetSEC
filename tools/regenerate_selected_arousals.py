#!/usr/bin/env python3
"""
Regenerate TargetSEC converted audios for selected arousal classes (e.g., 1,4,7),
and write metadata that maps each generated file back to the GT wav filename.

Usage examples:
  # regenerate ALL Test1 utterances, only classes 1/4/7
  python regenerate_selected_arousals.py \
    --save_root eval_outputs/regenerate_test1_gs07_sel147 \
    --classes 1 4 7

  # regenerate ONLY one specific GT file
  python regenerate_selected_arousals.py \
    --save_root eval_outputs/regenerate_one_0003_0442 \
    --classes 1 4 7 \
    --target_wav MSP-PODCAST-0003-0442.wav
"""

import os
import json
import ast
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

from src.Style.style_encoder import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.synthesizer_style_module import SynthesizerLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.diffusion_module_fixed import DiffusionLightningModule
from src.decoder.decoder_modules import broadcast_embeddings


# -----------------------------
# Default paths (match your setup)
# -----------------------------
DEFAULT_AUDIO_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/Audio"
DEFAULT_META_TEST = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"
DEFAULT_EMO_DIR   = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/avgclass_emo_embeds"

DEFAULT_CHECKPOINT_SYNTH = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/checkpoints_synthesizer/synthesizer_training_speakr-12-14_15-51-55-epoch=122-val_loss=17.55.ckpt"
DEFAULT_CHECKPOINT_LDM   = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/checkpoints/768_emo_256_speaker_diffusion_model_training-12-20_14-58-39-latest.ckpt"
DEFAULT_PRETRAINED_STYLE = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"

# -----------------------------
# Model configs (as in your benchmark script)
# -----------------------------
CONFIG_SYNTH = {
    "cross_attention_dim": 768,
    "generator": {
        "input_dim": 768,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        "upsample_rates": [5, 4, 4, 2, 2],
        "upsample_initial_channel": 1024,
        "upsample_kernel_sizes": [11, 8, 8, 4, 4],
        "gin_channels": 0,
        "resblock": "1",
    },
    "data": {
        "sampling_rate": 16000,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": None,
    },
    "training": {"learning_rate": 1e-4, "batch_size": 8},
}

CONFIG_LDM = {
    "training": {"learning_rate": 3e-5, "batch_size": 32, "cfg_prob": 0.3, "warmup_steps": 55_000},
    "inference": {"guidance_scale": 4.0, "guidance_rescale": 0.7},
    "cross_attention_dim": 768,
}


# -----------------------------
# Dataset that exposes GT filename + speaker + HuBERT
# -----------------------------
@dataclass
class Item:
    wav_name: str
    wav_path: str
    hubert: torch.Tensor
    speaker_emb: torch.Tensor


class Test1MetaDataset(Dataset):
    """
    Reads your parsed test1.txt (ast.literal_eval lines) and yields:
      - wav_name (e.g., MSP-PODCAST-0003-0442.wav)
      - wav_path (absolute)
      - hubert token ids (LongTensor, 1D)
      - speaker embedding (FloatTensor, 512)
    """
    def __init__(self, meta_path: str, audio_dir: str, target_wav: Optional[str] = None):
        self.audio_dir = audio_dir
        self.items: List[Item] = []

        with open(meta_path, "r") as f:
            for line in f:
                data = ast.literal_eval(line.strip())
                wav_name = os.path.basename(data["audio"])
                if not wav_name.endswith(".wav"):
                    wav_name = wav_name + ".wav"

                if target_wav is not None and wav_name != target_wav:
                    continue

                wav_path = os.path.join(audio_dir, wav_name)
                if not os.path.exists(wav_path):
                    # skip missing audio on disk
                    continue

                hubert = torch.tensor([int(x) for x in data["hubert"].split()], dtype=torch.long)
                speaker_emb = torch.tensor(data["spkr_embeds"], dtype=torch.float32)

                self.items.append(Item(wav_name=wav_name, wav_path=wav_path, hubert=hubert, speaker_emb=speaker_emb))

        # deterministic order
        self.items.sort(key=lambda x: x.wav_name)

        if target_wav is not None and len(self.items) == 0:
            raise FileNotFoundError(f"target_wav='{target_wav}' not found in meta or audio_dir.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "wav_name": it.wav_name,
            "wav_path": it.wav_path,
            "hubert": it.hubert,
            "speaker_emb": it.speaker_emb,
        }


def collate_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch size 1 recommended (keeps things simple & avoids padding wav itself)
    wav_name = [b["wav_name"] for b in batch]
    wav_path = [b["wav_path"] for b in batch]
    huberts = [b["hubert"] for b in batch]
    speaker_emb = torch.stack([b["speaker_emb"] for b in batch], dim=0)

    packed_huberts = pack_sequence(huberts, enforce_sorted=False)

    return {
        "wav_name": wav_name,
        "wav_path": wav_path,
        "hubert": packed_huberts,
        "speaker_emb": speaker_emb,
    }


# -----------------------------
# Emotion embedding bank
# -----------------------------
def load_emotion_embeddings_split(embedding_dir: str, split: str = "Test1") -> Dict[int, np.ndarray]:
    out = {}
    split_path = os.path.join(embedding_dir, split)
    for c in range(1, 8):
        out[c] = np.load(os.path.join(split_path, f"{c}.npy"))
    return out


def build_emo_bank(emotion_embeddings: Dict[int, np.ndarray], device: torch.device) -> torch.Tensor:
    emo_bank = torch.stack([torch.tensor(emotion_embeddings[c]) for c in range(1, 8)], dim=0).to(device)
    if emo_bank.ndim == 2:
        emo_bank = emo_bank.unsqueeze(1)  # (7,1,D)
    return emo_bank


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 4, 7], help="Target classes to generate (subset of 1..7)")
    parser.add_argument("--audio_dir", type=str, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--meta_test", type=str, default=DEFAULT_META_TEST)
    parser.add_argument("--emo_dir", type=str, default=DEFAULT_EMO_DIR)

    parser.add_argument("--checkpoint_synth", type=str, default=DEFAULT_CHECKPOINT_SYNTH)
    parser.add_argument("--checkpoint_ldm", type=str, default=DEFAULT_CHECKPOINT_LDM)
    parser.add_argument("--pretrained_style", type=str, default=DEFAULT_PRETRAINED_STYLE)

    parser.add_argument("--target_wav", type=str, default=None, help="If set, regenerate only this GT wav (e.g., MSP-PODCAST-0003-0442.wav)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of utterances to process")
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()

    # sanitize classes
    classes = sorted(set(args.classes))
    for c in classes:
        if c < 1 or c > 7:
            raise ValueError(f"Invalid class {c}. Must be in 1..7")

    pl.seed_everything(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_root, exist_ok=True)
    meta_path = os.path.join(args.save_root, "metadata.jsonl")

    wav_root = os.path.join(args.save_root, "wav")
    os.makedirs(wav_root, exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(wav_root, f"class_{c}"), exist_ok=True)

    sr = CONFIG_SYNTH["data"]["sampling_rate"]

    # emotion bank (7,1,D)
    emo_np = load_emotion_embeddings_split(args.emo_dir, split="Test1")
    emo_bank = build_emo_bank(emo_np, device=device)

    # load models
    pretrained_style_encoder = MelStyleEncoder(style_config)
    pretrained_style_encoder.load_state_dict(torch.load(args.pretrained_style, map_location="cpu"))
    pretrained_style_encoder = pretrained_style_encoder.to(device).eval()

    gen = Generator(CONFIG_SYNTH)
    discrim = MultiPeriodDiscriminator()

    synthesizer = SynthesizerLightningModule.load_from_checkpoint(
        args.checkpoint_synth,
        style_encoder=pretrained_style_encoder,
        decoder=gen,
        discriminator=discrim,
        config=CONFIG_SYNTH,
    ).to(device).eval()

    ldm = DiffusionLightningModule.load_from_checkpoint(
        args.checkpoint_ldm,
        style_encoder=pretrained_style_encoder,
        config=CONFIG_LDM,
    ).to(device).eval()

    for p in synthesizer.parameters():
        p.requires_grad = False
    for p in ldm.parameters():
        p.requires_grad = False

    decoder = synthesizer.decoder.to(device).eval()
    dict_proj = synthesizer.dict.to(device).eval()

    # dataloader with filenames
    ds = Test1MetaDataset(meta_path=args.meta_test, audio_dir=args.audio_dir, target_wav=args.target_wav)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_meta)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Found {len(ds)} items (target_wav={args.target_wav})")
    print(f"[INFO] Generating classes: {classes}")
    print(f"[INFO] Saving to: {args.save_root}")

    # map from class -> index in 0..6
    class_to_i = {c: (c - 1) for c in range(1, 8)}

    processed = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dl):
            if args.limit is not None and processed >= args.limit:
                break

            # unpack
            wav_name = batch["wav_name"][0]
            wav_path = batch["wav_path"][0]

            linguistic_packed = batch["hubert"]
            speaker = batch["speaker_emb"].to(device)

            # (B=1, T, dim_hubert_tokens) after pad
            linguistic, lengths = torch.nn.utils.rnn.pad_packed_sequence(linguistic_packed, batch_first=True)
            linguistic = linguistic.to(device)

            # match your pipeline
            linguistic = dict_proj(linguistic).transpose(1, 2)  # (1, C, T') depending on dict

            # We'll generate for selected classes only (not all 7)
            for c in classes:
                i = class_to_i[c]  # 0..6

                # build K=1 conditioning for this class
                linguistic_k = linguistic  # (1, ...)
                speaker_k = speaker        # (1,512)
                emo_k = emo_bank[i:i+1]    # (1,1,D)

                style = ldm(emo_k.float(), speaker_k.float())
                if style.ndim == 3:
                    style = style.squeeze(1)

                emb = broadcast_embeddings(linguistic_k, speaker_k, style)
                y_hat = decoder(emb).squeeze(1)  # (1,T_audio) or (T_audio,)
                if y_hat.dim() == 2:
                    y_hat = y_hat[0]

                y_hat = y_hat.clamp(-1.0, 1.0)

                # save wav using GT name to keep traceability
                base = os.path.splitext(wav_name)[0]
                out_name = f"{base}__class{c}.wav"
                out_path = os.path.join(wav_root, f"class_{c}", out_name)
                torchaudio.save(out_path, y_hat.detach().cpu().unsqueeze(0), sample_rate=sr)

                rec = {
                    "gt_wav_name": wav_name,
                    "gt_wav_path": wav_path,
                    "class": int(c),
                    "generated_wav_path": out_path,
                    "checkpoint_ldm": args.checkpoint_ldm,
                    "checkpoint_synth": args.checkpoint_synth,
                }
                with open(meta_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            processed += 1
            if processed % 200 == 0:
                print(f"[progress] processed {processed}/{len(ds)}")

    print("[DONE]")
    print(f"Saved WAVs to: {wav_root}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
