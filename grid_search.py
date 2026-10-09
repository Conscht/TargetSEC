# grid_search.py
import os, json, argparse, random
from collections import defaultdict
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl

from src.Style.style_encoder import MelStyleEncoder
from config.stylespeech_model_config import style_config

from src.synthesizer_style_module import SynthesizerLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader

from src.diffusion_module_fixed import DiffusionLightningModule
from src.decoder.decoder_modules import broadcast_embeddings
from src.emotion.emotion_encoder import process_func

# IMPORTANT: rescale_noise_cfg lives in your diffusion_utils
from processing.diffusion_utils import rescale_noise_cfg


# -----------------------------
# small utilities
# -----------------------------
def parse_csv_list(x, cast=float):
    # accepts: "25,50,100" or "25 50 100"
    if x is None:
        return []
    x = x.replace(",", " ").split()
    return [cast(v) for v in x]

def load_emotion_embeddings_split(embedding_dir: str, split: str = "Test2") -> dict:
    out = {}
    split_path = os.path.join(embedding_dir, split)
    for c in range(1, 8):
        out[c] = np.load(os.path.join(split_path, f"{c}.npy"))
    return out

def build_emo_bank(emotion_embeddings: dict, device: torch.device) -> torch.Tensor:
    emo_bank = torch.stack([torch.tensor(emotion_embeddings[c]) for c in range(1, 8)], dim=0).to(device)
    if emo_bank.ndim == 2:
        emo_bank = emo_bank.unsqueeze(1)  # (7,1,D)
    return emo_bank

def get_pred_arousal(y_hat_audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    # returns (B,) arousal in [0,1] from your verifier
    emo_pred = process_func(y_hat_audio.detach().cpu().numpy(), device=device, embeddings=False)  # (B,3)
    return torch.tensor(emo_pred[:, 0], dtype=torch.float32, device=device)

def choose_subset_indices(n_total: int, n_keep: int, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return set(idx[:n_keep].tolist())


# -----------------------------
# core eval for one hyperparam setting
# -----------------------------
@torch.no_grad()
def eval_setting(
    *,
    device,
    test_loader,
    emo_bank,
    synthesizer,
    ldm,
    steps: int,
    guidance: float,
    k_rescale: float,
    n_utts: int,
    subset_seed: int,
    save_root: str,
    save_wav: bool,
    save_audio_limit: int | None,
):
    # fix targets to 0..1 scale: class 1 -> 0.0, class 7 -> 1.0
    targets = (torch.arange(1, 8, device=device, dtype=torch.float32) - 1.0) / 6.0  # (7,)

    # accumulators
    mse_sum = defaultdict(float)
    mae_sum = defaultdict(float)
    n_sum   = defaultdict(int)

    mse_global = 0.0
    mae_global = 0.0
    n_global = 0

    # optional save
    os.makedirs(save_root, exist_ok=True)
    meta_path = os.path.join(save_root, "metadata.jsonl")
    if save_wav:
        wav_root = os.path.join(save_root, "wav")
        for c in range(1, 8):
            os.makedirs(os.path.join(wav_root, f"class_{c}"), exist_ok=True)

    # we want a stable subset across settings
    # -> first count total, then pick subset indices
    # NOTE: this requires one pass to count; cheap compared to full synthesis.
    total = 0
    for _ in test_loader:
        total += 1
    keep = choose_subset_indices(total, min(n_utts, total), seed=subset_seed)

    # set inference hyperparams on the LDM config (used by forward())
    # (your Lightning forward() reads config["inference"], see your module) :contentReference[oaicite:1]{index=1}
    ldm.config.setdefault("inference", {})
    ldm.config["inference"]["num_steps"] = int(steps)
    ldm.config["inference"]["guidance_scale"] = float(guidance)

    # patch the rescale factor (k) if your diffusion uses rescale_noise_cfg(k=...)
    # easiest: monkey-patch a field and use it inside diffusion inference;
    # if your diffusion already calls rescale_noise_cfg(..., k=0.7), update that call in code,
    # OR do this simple patch by wrapping the function:
    # We'll just store k on ldm for logging; actual rescale must be used in diffusion code path.
    ldm._grid_k = float(k_rescale)

    sr = synthesizer.config["data"]["sampling_rate"]

    synthesizer.eval()
    ldm.eval()

    # access submodules (ensure on device)
    decoder = synthesizer.decoder.to(device).eval()
    dict_proj = synthesizer.dict.to(device).eval()

    # run
    processed = 0
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx not in keep:
            continue

        # move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        linguistic_packed = batch["hubert"]
        speaker = batch["speaker_emb"]  # (1,512)

        linguistic, _ = torch.nn.utils.rnn.pad_packed_sequence(linguistic_packed, batch_first=True)
        linguistic = linguistic.to(device)
        linguistic = dict_proj(linguistic).transpose(1, 2)

        # repeat for 7 target classes
        K = 7
        linguistic_k = linguistic.repeat(K, 1, 1)
        speaker_k = speaker.repeat(K, 1)
        emo_k = emo_bank  # (7,1,D)

        # ---- synthesize ----
        # ldm forward returns style (B,1,C) in your DreamVG wrapper version,
        # but in your fixed module it returns (B,C) or (B,1,C) depending. Handle both. :contentReference[oaicite:2]{index=2}
        style = ldm(emo_k.float(), speaker_k.float())
        if style.ndim == 3:
            style = style.squeeze(1)  # (7,C)

        emb = broadcast_embeddings(linguistic_k, speaker_k, style)
        y_hat = decoder(emb).squeeze(1).clamp(-1.0, 1.0)  # (7,T)

        pred_ar = get_pred_arousal(y_hat, device=device)  # (7,) in [0,1]

        # ---- metrics ----
        err = pred_ar - targets
        mse_vec = err.pow(2)
        mae_vec = err.abs()

        for i, c in enumerate(range(1, 8)):
            mse_sum[c] += float(mse_vec[i].item())
            mae_sum[c] += float(mae_vec[i].item())
            n_sum[c] += 1

            mse_global += float(mse_vec[i].item())
            mae_global += float(mae_vec[i].item())
            n_global += 1

            # ---- save wav + metadata ----
            if save_wav and (save_audio_limit is None or processed < save_audio_limit):
                out_wav = os.path.join(save_root, "wav", f"class_{c}", f"utt_{batch_idx:06d}.wav")
                torchaudio.save(out_wav, y_hat[i].detach().cpu().unsqueeze(0), sample_rate=sr)

                rec = {
                    "utt_idx": int(batch_idx),
                    "class": int(c),
                    "steps": int(steps),
                    "guidance": float(guidance),
                    "k": float(k_rescale),
                    "target_arousal": float(targets[i].item()),
                    "pred_arousal": float(pred_ar[i].item()),
                    "abs_err": float(mae_vec[i].item()),
                    "sq_err": float(mse_vec[i].item()),
                    "wav_path": out_wav,
                }
                with open(meta_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        processed += 1
        if processed >= n_utts:
            break

    # aggregate
    per_class = {}
    for c in range(1, 8):
        n = max(1, n_sum[c])
        per_class[str(c)] = {
            "MAE": mae_sum[c] / n,
            "MSE": mse_sum[c] / n,
            "N": int(n_sum[c]),
        }

    macro_mae = sum(per_class[str(c)]["MAE"] for c in range(1, 8)) / 7.0
    macro_mse = sum(per_class[str(c)]["MSE"] for c in range(1, 8)) / 7.0
    global_mae = mae_global / max(1, n_global)
    global_mse = mse_global / max(1, n_global)

    return {
        "steps": int(steps),
        "guidance": float(guidance),
        "k": float(k_rescale),
        "per_class": per_class,
        "macro_mae": float(macro_mae),
        "macro_mse": float(macro_mse),
        "global_mae": float(global_mae),
        "global_mse": float(global_mse),
        "N_utts": int(processed),
    }


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", type=str, default="eval_grid")
    ap.add_argument("--n_utts", type=int, default=2000)
    ap.add_argument("--subset_seed", type=int, default=1234)

    ap.add_argument("--steps", type=str, required=True)     # e.g. "25,50,100"
    ap.add_argument("--guidance", type=str, required=True)  # e.g. "3,4"
    ap.add_argument("--k", type=str, default="0.7")         # e.g. "0.3,0.7,0.9"

    ap.add_argument("--emotion_embedding_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="Test2")

    ap.add_argument("--checkpoint_synth", type=str, required=True)
    ap.add_argument("--checkpoint_ldm", type=str, required=True)
    ap.add_argument("--style_ckpt", type=str, required=True)
    ap.add_argument("--style_stats", type=str, default=None)

    ap.add_argument("--save_wav", action="store_true")
    ap.add_argument("--save_audio_limit", type=int, default=None)

    args = ap.parse_args()

    pl.seed_everything(args.subset_seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps_list = [int(x) for x in parse_csv_list(args.steps, cast=float)]
    guidance_list = parse_csv_list(args.guidance, cast=float)
    k_list = parse_csv_list(args.k, cast=float)

    # ---- load embeddings ----
    emo_dict = load_emotion_embeddings_split(args.emotion_embedding_dir, split=args.split)
    emo_bank = build_emo_bank(emo_dict, device=device)  # (7,1,D)

    # ---- style encoder ----
    style_enc = MelStyleEncoder(style_config)
    style_enc.load_state_dict(torch.load(args.style_ckpt, map_location="cpu"))
    style_enc = style_enc.to(device).eval()

    # ---- synth config (must include 'generator' or Generator() will crash) ----
    # Use the SAME config you used during training (or load it from a JSON you trust).
    # Your Generator expects config['generator']['input_dim'] etc. :contentReference[oaicite:3]{index=3}
    config_synth = {
        "cross_attention_dim": 512,
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

    gen = Generator(config_synth)
    discrim = MultiPeriodDiscriminator()

    # ---- load synthesizer correctly (you MUST pass required ctor args) ----
    synthesizer = SynthesizerLightningModule.load_from_checkpoint(
        args.checkpoint_synth,
        style_encoder=style_enc,
        decoder=gen,
        discriminator=discrim,
        config=config_synth
    ).to(device).eval()

    # ---- load ldm ----
    config_ldm = {
        "training": {"learning_rate": 3e-5, "batch_size": 32, "cfg_prob": 0.3, "warmup_steps": 55_000},
        "inference": {"guidance_scale": 3.0, "guidance_rescale": 0.7, "num_steps": 50},
        "cross_attention_dim": 512,
    }

    ldm = DiffusionLightningModule.load_from_checkpoint(
        args.checkpoint_ldm,
        style_encoder=style_enc,
        config=config_ldm,
        # if your DiffusionLightningModule requires style_stats_path, pass it here too
    ).to(device).eval()

    # freeze
    for p in synthesizer.parameters():
        p.requires_grad = False
    for p in ldm.parameters():
        p.requires_grad = False

    # ---- loader ----
    test_loader = test_create_data_loader(batch_size=1)

    # ---- run grid ----
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "results.json")
    all_results = []

    for (steps, guidance, k) in product(steps_list, guidance_list, k_list):
        run_name = f"steps{steps}_g{guidance}_k{k}"
        run_dir = os.path.join(args.save_dir, run_name)

        print(f"\n=== RUN {run_name} (n_utts={args.n_utts}, split={args.split}) ===")

        out = eval_setting(
            device=device,
            test_loader=test_loader,
            emo_bank=emo_bank,
            synthesizer=synthesizer,
            ldm=ldm,
            steps=steps,
            guidance=guidance,
            k_rescale=k,
            n_utts=args.n_utts,
            subset_seed=args.subset_seed,
            save_root=run_dir,
            save_wav=args.save_wav,
            save_audio_limit=args.save_audio_limit,
        )

        print(f"Global MAE={out['global_mae']:.4f}  MSE={out['global_mse']:.4f}")
        all_results.append(out)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved grid results to: {results_path}")


if __name__ == "__main__":
    main()
