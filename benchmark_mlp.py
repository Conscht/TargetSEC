import os
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl

# --- IMPORTS ---
from src.Style.style_encoder import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.synthesizer_style_module import SynthesizerLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
from src.decoder.decoder_modules import broadcast_embeddings
from src.emotion.emotion_encoder import process_func

# 🔹 NEW: Import your MLP Module
from Ablation.style_emo_mlp_module import StyleEmoLightningModule

# -----------------------------
# USER CONFIG
# -----------------------------
# Set this to match the checkpoint you are loading!
USE_SPEAKER_COND = False  # True = Emo+Spk MLP, False = Emo-Only MLP

# Paths
emotion_embedding_dir = r"/sc/home/constantin.auga/New folder/Audio/MSP-Podcast-1.10/avgclass_emo_embeds"
checkpoint_synth = r"/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/checkpoints_synthesizer/synthesizer_training_speakr-12-14_15-51-55-epoch=122-val_loss=17.55.ckpt"

# 🔹 PATH TO YOUR NEW MLP CHECKPOINT
# (Paste the path to your .ckpt file from 'checkpoints_ablation_mlp' here)
checkpoint_mlp = r"/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/checkpoints_ablation_mlp/MLP_Baseline_noSpk_01-22_16-11-epoch=299-val_loss=0.564.ckpt"

SAVE_ROOT = f"eval_outputs/test1_MLP_ABLAT_spk{USE_SPEAKER_COND}"   # Auto-rename output folder
SAVE_WAV = True
SAVE_AUDIO_LIMIT = None  # Set to 100 for a quick test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configs (MLP config is minimal)
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
}

# -----------------------------
# Helpers
# -----------------------------
def load_emotion_embeddings_split(embedding_dir: str, split: str = "Test1") -> dict:
    import numpy as np
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
    emo_pred = process_func(
        y_hat_audio.detach().cpu().numpy(),
        device=device,
        embeddings=False
    )
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

    # ---- load embeddings ----
    emotion_embeddings = load_emotion_embeddings_split(emotion_embedding_dir, split="Test1")
    emo_bank = build_emo_bank(emotion_embeddings, device=device)

    targets = (torch.arange(1, 8, device=device, dtype=torch.float32) - 1.0) / 6.0

    # ---- load models ----
    # 1. Style Encoder
    pretrained_style_encoder = MelStyleEncoder(style_config)
    pretrained_style_encoder.load_state_dict(torch.load(
        "/sc/home/constantin.auga/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style",
        map_location="cpu"
    ))
    pretrained_style_encoder = pretrained_style_encoder.to(device).eval()

    # 2. Synthesizer (Decoder)
    gen = Generator(config_synth)
    discrim = MultiPeriodDiscriminator()
    synthesizer = SynthesizerLightningModule.load_from_checkpoint(
        checkpoint_synth,
        style_encoder=pretrained_style_encoder,
        decoder=gen,
        discriminator=discrim,
        config=config_synth
    ).to(device).eval()
    
    decoder = synthesizer.decoder.to(device).eval()
    dict_proj = synthesizer.dict.to(device).eval()

    # 3. 🔹 MLP MODEL (Replaces LDM)
    print(f"Loading MLP Model (Speaker Cond: {USE_SPEAKER_COND})...")
    
    # We pass a minimal config since inference only needs dimensions
    mlp_config = {"training": {"learning_rate": 1e-4}} 
    
    mlp_model = StyleEmoLightningModule.load_from_checkpoint(
        checkpoint_mlp,
        style_encoder=pretrained_style_encoder,
        config=mlp_config,
        use_speaker_cond=USE_SPEAKER_COND,
        # Ensure this path is correct or passed relative to script location
        style_stats_path="/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/style_stats_new.pt"
    ).to(device).eval()

    # Freeze everything
    for p in synthesizer.parameters(): p.requires_grad = False
    for p in mlp_model.parameters(): p.requires_grad = False

    # ---- metrics accumulators ----
    mse_sum = defaultdict(float)
    mae_sum = defaultdict(float)
    n_sum   = defaultdict(int)
    mse_global_sum = 0.0
    mae_global_sum = 0.0
    n_global = 0

    # ---- loader ----
    test_loader = test_create_data_loader(batch_size=1)

    print(f"Starting Inference on {device}...")

    # ---- eval loop ----
    for batch_idx, batch in enumerate(test_loader):
        if SAVE_AUDIO_LIMIT is not None and batch_idx >= SAVE_AUDIO_LIMIT:
            break

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        linguistic_packed = batch["hubert"]
        speaker = batch["speaker_emb"]  # (1,512)

        linguistic, _ = torch.nn.utils.rnn.pad_packed_sequence(linguistic_packed, batch_first=True)
        linguistic = linguistic.to(device)
        linguistic = dict_proj(linguistic).transpose(1, 2)

        # Prepare Batch for all 7 classes
        K = 7
        linguistic_k = linguistic.repeat(K, 1, 1)  # (7,*,*)
        speaker_k = speaker.repeat(K, 1)           # (7,512)
        emo_k = emo_bank                           # (7,1,D)

        with torch.inference_mode():
            # 🔹 RUN MLP INFERENCE
            # The MLP forward handles normalization/denormalization internally
            # It expects (B, 1024) and optional (B, 512)
            
            style = mlp_model(emo_k.float(), speaker_k.float())
            
            # MLP returns (B, 128), we need to ensure shape match for broadcaster
            if style.ndim == 3: style = style.squeeze(1)

            # Generate Audio
            emb = broadcast_embeddings(linguistic_k, speaker_k, style)
            y_hat = decoder(emb).squeeze(1)  # (7,T_audio)

            y_hat = y_hat.clamp(-1.0, 1.0)
            
            # Check Arousal
            pred_ar = get_pred_arousal(y_hat, device=device)

        # Compute MAE/MSE
        err = pred_ar - targets
        mse_vec = err.pow(2)
        mae_vec = err.abs()

        for i, emotion_class in enumerate(range(1, 8)):
            mse_sum[emotion_class] += float(mse_vec[i].item())
            mae_sum[emotion_class] += float(mae_vec[i].item())
            n_sum[emotion_class]   += 1
            mse_global_sum += float(mse_vec[i].item())
            mae_global_sum += float(mae_vec[i].item())
            n_global += 1

            if SAVE_WAV:
                out_path = os.path.join(SAVE_ROOT, "wav", f"class_{emotion_class}", f"utt_{batch_idx:06d}.wav")
                torchaudio.save(out_path, y_hat[i].detach().cpu().unsqueeze(0), sample_rate=sr)
                
                # Metadata log
                rec = {
                    "batch_idx": batch_idx,
                    "class": emotion_class,
                    "target_arousal": float(targets[i].item()),
                    "pred_arousal": float(pred_ar[i].item()),
                    "abs_err": float(mae_vec[i].item()),
                    "sq_err": float(mse_vec[i].item()),
                    "model_type": "MLP",
                    "use_speaker_cond": USE_SPEAKER_COND
                }
                with open(meta_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        if batch_idx % 50 == 0 and batch_idx > 0:
            print(f"[progress] processed {batch_idx} utterances")

    # ---- Print results ----
    print(f"\n=== MLP Ablation Results (Speaker Cond: {USE_SPEAKER_COND}) ===")
    per_class_mae = {}
    for c in range(1, 8):
        n = max(1, n_sum[c])
        per_class_mae[c] = mae_sum[c] / n
        print(f"Class {c}: MAE={per_class_mae[c]:.4f}")

    global_mae = mae_global_sum / max(1, n_global)
    print(f"\nGlobal MAE: {global_mae:.4f}")