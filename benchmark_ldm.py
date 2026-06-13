import os
import json
from collections import defaultdict

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


# -----------------------------
# Paths / config
# -----------------------------
emotion_embedding_dir = r"/sc/home/constantin.auga/New folder/Audio/MSP-Podcast-1.10/avgclass_emo_embeds"

checkpoint_synth = r"/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/checkpoints_synthesizer/synthesizer_training_speakr-12-14_15-51-55-epoch=122-val_loss=17.55.ckpt"
checkpoint_ldm   = r"/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/checkpoints_ablation_ldm/LDM_Ablation_wSpk_01-23_17-36-epoch=586-val_loss=0.47.ckpt"

# -----------------------------
# NEW: Ablation flag
# True  = LDM conditioned on emotion + speaker (full)
# False = LDM conditioned on emotion only (speaker condition nulled/zeroed inside module)
# -----------------------------
USE_SPEAKER_COND = True

# -----------------------------
# Output settings
# -----------------------------
mode = "emo_spk" if USE_SPEAKER_COND else "emo_only"
SAVE_ROOT = f"eval_outputs/test1_{mode}_768crossatt_synth_long_final_eval_guidance4__gs09_guidance09"
SAVE_WAV = True
SAVE_AUDIO_LIMIT = None  # None = save all, or int (e.g. 200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_synth = {
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

# -----------------------------
# IMPORTANT: fix key name to match DiffusionLightningModule.forward()
# it uses "guidance_rescale_k"
# -----------------------------
config_ldm = {
    "training": {"learning_rate": 3e-5, "batch_size": 32, "cfg_prob": 0.3, "warmup_steps": 55_000},
    "inference": {"guidance_scale": 4.0, "guidance_rescale_k": 0.9},  # <-- fixed key
    "cross_attention_dim": 768,
}





# -----------------------------
# Helpers
# -----------------------------
def load_emotion_embeddings_split(embedding_dir: str, split: str = "Test1") -> dict:
    """
    Loads avg-class emotion embeddings from embedding_dir/<split>/{1..7}.npy
    Returns dict[int -> np.ndarray]
    """
    import numpy as np
    out = {}
    split_path = os.path.join(embedding_dir, split)
    for c in range(1, 8):
        out[c] = np.load(os.path.join(split_path, f"{c}.npy"))
    return out


def build_emo_bank(emotion_embeddings: dict, device: torch.device) -> torch.Tensor:
    """
    Returns emo_bank of shape (7,1,D) for classes 1..7
    """
    emo_bank = torch.stack([torch.tensor(emotion_embeddings[c]) for c in range(1, 8)], dim=0).to(device)
    if emo_bank.ndim == 2:
        emo_bank = emo_bank.unsqueeze(1)  # (7,1,D)
    return emo_bank


def get_pred_arousal(y_hat_audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    y_hat_audio: (B, T)
    returns: (B,) arousal predictions in [0,1] from the emotion verifier
    """
    emo_pred = process_func(
        y_hat_audio.detach().cpu().numpy(),
        device=device,
        embeddings=False
    )  # expected (B,3)
    return torch.tensor(emo_pred[:, 0], dtype=torch.float32, device=device)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    pl.seed_everything(1234)

    # ---- output dirs ----
    os.makedirs(SAVE_ROOT, exist_ok=True)
    meta_path = os.path.join(SAVE_ROOT, "metadata.jsonl")
    if SAVE_WAV:
        wav_root = os.path.join(SAVE_ROOT, "wav")
        for c in range(1, 8):
            os.makedirs(os.path.join(wav_root, f"class_{c}"), exist_ok=True)

    sr = config_synth["data"]["sampling_rate"]

    # ---- load embeddings (Test1) ----
    emotion_embeddings = load_emotion_embeddings_split(emotion_embedding_dir, split="Test1")
    emo_bank = build_emo_bank(emotion_embeddings, device=device)  # (7,1,D)

    # Targets on 0..1 scale: class 1 -> 0.0, class 7 -> 1.0
    targets = (torch.arange(1, 8, device=device, dtype=torch.float32) - 1.0) / 6.0  # (7,)

    # ---- load style encoder ----
    pretrained_style_encoder = MelStyleEncoder(style_config)
    pretrained_style_encoder.load_state_dict(torch.load(
        "/sc/home/constantin.auga/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style",
        map_location="cpu"
    ))
    pretrained_style_encoder = pretrained_style_encoder.to(device).eval()

    # ---- load synthesizer ----
    gen = Generator(config_synth)
    discrim = MultiPeriodDiscriminator()

    synthesizer = SynthesizerLightningModule.load_from_checkpoint(
        checkpoint_synth,
        style_encoder=pretrained_style_encoder,
        decoder=gen,
        discriminator=discrim,
        config=config_synth
    ).to(device).eval()

    # ---- load diffusion model with ablation flag ----
    # NOTE: This requires DiffusionLightningModule.__init__ to accept use_speaker_cond
    # and implement emotion-only by nulling/zeroing speaker token while keeping cond dim fixed.
    ldm = DiffusionLightningModule.load_from_checkpoint(
        checkpoint_ldm,
        style_encoder=pretrained_style_encoder,
        config=config_ldm,
        use_speaker_cond=USE_SPEAKER_COND,
    ).to(device).eval()

    # freeze
    for p in synthesizer.parameters():
        p.requires_grad = False
    for p in ldm.parameters():
        p.requires_grad = False

    decoder = synthesizer.decoder.to(device).eval()
    dict_proj = synthesizer.dict.to(device).eval()

    # ---- metrics accumulators ----
    mse_sum = defaultdict(float)
    mae_sum = defaultdict(float)
    n_sum   = defaultdict(int)

    mse_global_sum = 0.0
    mae_global_sum = 0.0
    n_global = 0

    # ---- loader ----
    test_loader = test_create_data_loader(batch_size=1)

    # ---- eval loop ----
    for batch_idx, batch in enumerate(test_loader):
        if SAVE_AUDIO_LIMIT is not None and batch_idx >= SAVE_AUDIO_LIMIT:
            break

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        linguistic_packed = batch["hubert"]      # PackedSequence
        speaker = batch["speaker_emb"]           # (1,512) typically

        linguistic, _ = torch.nn.utils.rnn.pad_packed_sequence(linguistic_packed, batch_first=True)  # (1,T,dim)
        linguistic = linguistic.to(device)
        linguistic = dict_proj(linguistic).transpose(1, 2)

        # Repeat for all 7 classes
        K = 7
        linguistic_k = linguistic.repeat(K, 1, 1)  # (7,*,*)
        speaker_k = speaker.repeat(K, 1)           # (7,512)
        emo_k = emo_bank                           # (7,1,D)

        with torch.inference_mode():
            style = ldm(emo_k.float(), speaker_k.float())
            if style.ndim == 3:
                style = style.squeeze(1)

            emb = broadcast_embeddings(linguistic_k, speaker_k, style)
            y_hat = decoder(emb).squeeze(1)  # (7,T_audio)

            y_hat = y_hat.clamp(-1.0, 1.0)
            pred_ar = get_pred_arousal(y_hat, device=device)  # (7,) in [0,1]

        # Compute MAE/MSE on 0..1 scale
        err = pred_ar - targets
        mse_vec = err.pow(2)
        mae_vec = err.abs()

        # accumulate per class + global pooled
        for i, emotion_class in enumerate(range(1, 8)):
            mse_sum[emotion_class] += float(mse_vec[i].item())
            mae_sum[emotion_class] += float(mae_vec[i].item())
            n_sum[emotion_class]   += 1

            mse_global_sum += float(mse_vec[i].item())
            mae_global_sum += float(mae_vec[i].item())
            n_global += 1

            # Save wav + metadata (so you can rerun WV-MOS later)
            if SAVE_WAV:
                out_path = os.path.join(SAVE_ROOT, "wav", f"class_{emotion_class}", f"utt_{batch_idx:06d}.wav")
                torchaudio.save(out_path, y_hat[i].detach().cpu().unsqueeze(0), sample_rate=sr)

                rec = {
                    "batch_idx": batch_idx,
                    "class": emotion_class,
                    "wav_path": out_path,
                    "target_arousal": float(targets[i].item()),
                    "pred_arousal": float(pred_ar[i].item()),
                    "abs_err": float(mae_vec[i].item()),
                    "sq_err": float(mse_vec[i].item()),
                    "use_speaker_cond": bool(USE_SPEAKER_COND),
                    "checkpoint_ldm": checkpoint_ldm,
                    "checkpoint_synth": checkpoint_synth,
                }
                with open(meta_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        if batch_idx % 200 == 0 and batch_idx > 0:
            print(f"[progress] processed {batch_idx} utterances")

    # ---- Print results ----
    print("\n=== Test1 Arousal Goodness (0..1 scale) ===")
    print(f"USE_SPEAKER_COND = {USE_SPEAKER_COND}")
    per_class_mse = {}
    per_class_mae = {}

    for c in range(1, 8):
        n = max(1, n_sum[c])
        per_class_mse[c] = mse_sum[c] / n
        per_class_mae[c] = mae_sum[c] / n
        print(f"Class {c}: MAE={per_class_mae[c]:.4f}  MSE={per_class_mse[c]:.4f}  N={n_sum[c]}")

    macro_mae = sum(per_class_mae.values()) / 7.0
    macro_mse = sum(per_class_mse.values()) / 7.0

    global_mae = mae_global_sum / max(1, n_global)
    global_mse = mse_global_sum / max(1, n_global)

    print(f"\nGlobal pooled: MAE={global_mae:.4f}  MSE={global_mse:.4f}  N={n_global}")
    print(f"Macro avg:     MAE={macro_mae:.4f}  MSE={macro_mse:.4f}")

    if SAVE_WAV:
        print(f"\nSaved WAVs to: {os.path.join(SAVE_ROOT, 'wav')}")
        print(f"Saved metadata to: {meta_path}")
