#!/usr/bin/env python
# ---------------------------------------------------------------
#  compare_diffusion_latents_mysetup.py
# ---------------------------------------------------------------
#  * requires the same environment your project already uses
#  * adjust only the PATHS section
# ---------------------------------------------------------------
import os, glob, json, numpy as np
import torch, librosa
import torch.nn.functional as F
from tqdm import tqdm
# ----------------------------------------------------------------
#  ---- YOUR  OWN  MODULES  --------------------------------------
# ----------------------------------------------------------------
from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.emotion.emotion_encoder import EmotionModel
from src.diffusion_model_wholeseq import DiffusionLightningModule
# ----------------------------------------------------------------

# -----------------------  PATHS ---------------------------------
ROOT              = r"C:\Users\Conscht\Documents\New folder\Audio\MSP-Podcast-1.10"  # <-- change if needed
EMB_DIR           = f"{ROOT}/avgclass_emo_embeds/Train"               # 1.npy … 7.npy

AUDIO_PATH        = r"C:\Users\Conscht\Documents\New folder\Audio\Audio"
MEL_PATH          = r"C:\Users\Conscht\Documents\New folder\mel_spectograms\Train\MSP-PODCAST_0001_0049_mel.pt"

STYLE_WGHT        = "/Users/Conscht/Documents/New folder/Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
DIFFUSION_CKPT    = r"C:\Users\Conscht\Documents\New folder\Code\EmoConv-LDM\checkpoints\diffusion_model_training-11-16_01-00-29-epoch=98-val_loss=0.07.ckpt"

# -----------------------  DEVICE --------------------------------
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
print(f"[INFO] device = {dev}")
import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything(42)


# ================= 1.  LOAD  ENCODERS ===========================
style_enc = MelStyleEncoder(style_config).to(dev).eval()
style_enc.load_state_dict(torch.load(STYLE_WGHT, map_location=dev))

emotion_enc = EmotionModel.from_pretrained(
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
).to(dev).eval()

# dummy cfg for Lightning:
cfg = {"training": {"learning_rate": 1e-4, "batch_size": 16}}
diff_lm = DiffusionLightningModule.load_from_checkpoint(
            DIFFUSION_CKPT,
            config=cfg,
          ).to(dev).eval()

# ===== 2.  GROUND-TRUTH  LATENT  FROM  REFERENCE  MEL ===========
ref_mel = torch.load(MEL_PATH, map_location=dev)          # (T,80)
ref_mel = torch.tensor(ref_mel, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(dev)
latent_gt = style_enc(ref_mel).view(1, 128)               # (1,128)
print("[OK] ground-truth style latent computed\n")

# ----------  speaker vector (dummy 512-D zeros) -----------------
spk_emb = torch.zeros(1, 512, device=dev)                 # replace if availablere

# ----------  helper --------------------------------------------
def cos(a, b): return F.cosine_similarity(a, b, dim=1).mean().item()

# ================= 3.  LOOP  OVER  AROUSAL  CLASSES =============
npy_files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
if not npy_files:
    raise FileNotFoundError(f"No *.npy files in {EMB_DIR}")

print(f"{'class':>6} | {'MSE':>9} | {'cos-sim':>7}")
print("-"*28)


for npy in sorted(glob.glob(os.path.join(EMB_DIR, "*.npy"))):  # reversed
    cls = os.path.splitext(os.path.basename(npy))[0]
    emo_vec = torch.from_numpy(np.load(npy)).to(dev).unsqueeze(0)  # (1,1024)

    cond = torch.cat([emo_vec, spk_emb], dim=1).unsqueeze(1)       # (1, 1, 1536)

    # ⬇️ Reset RNG before inference
    seed_everything(42)

    latent_pred = diff_lm.diffusion_model.inference(
        cond, num_steps=100, guidance_scale=3.0
    ).view(1, 128)

    mse = F.mse_loss(latent_pred, latent_gt).item()
    cs = cos(latent_pred, latent_gt)

    print(f"{cls:>6} | {mse:9.4f} | {cs:7.3f}")