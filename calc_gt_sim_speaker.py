import os
import random
import csv
import numpy as np
import torch
import torchaudio
import logging
from tqdm import tqdm
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

# --- SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ECAPA CLASS
# ==========================================
class ECAPA:
    def __init__(self, device=device):
        print(f"Loading ECAPA model on {device}...")
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)},
        )

    @torch.inference_mode()
    def embed(self, wav_path: str):
        try:
            wav, sr = torchaudio.load(wav_path)
            if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            wav = wav.to(self.device)
            emb = self.model.encode_batch(wav).squeeze()
            return emb.detach().cpu()
        except Exception:
            return None

    def cosine(self, e1, e2):
        if e1 is None or e2 is None: return None
        return torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()

# ==========================================
# 2. FILTER FOR TEST1 ONLY
# ==========================================
def get_test_set_dict(csv_path, audio_dir):
    csv_path = Path(csv_path)
    audio_dir = Path(audio_dir)
    spk2wavs = {}
    
    logger.info(f"Loading metadata from {csv_path}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # DEBUG: Print columns to ensure we have the right name
        if reader.fieldnames:
            logger.info(f"CSV Columns found: {reader.fieldnames}")

        count = 0
        skipped_wrong_split = 0
        
        for row in reader:
            # 1. Check Split
            # Column name might be 'Split', 'Split_Set', or 'Partition' based on version
            # We try standard MSP keys
            split_val = row.get("Split", row.get("Split_Set", row.get("Partition", "")))
            
            # STRICT FILTER: Only accept Test1
            if split_val.strip() != "Test1":
                skipped_wrong_split += 1
                continue

            # 2. Get Filename & Speaker
            filename = row.get("FileName", row.get("name", "")) 
            spkr_id = row.get("SpkrID", row.get("speaker_id", ""))
            
            if not filename or not spkr_id: continue
            if not filename.endswith(".wav"): filename += ".wav"
            
            full_path = audio_dir / filename
            
            if full_path.exists():
                if spkr_id not in spk2wavs: spk2wavs[spkr_id] = []
                spk2wavs[spkr_id].append(str(full_path))
                count += 1

    # Filter for speakers with at least 2 clips in Test1
    valid_speakers = {k: v for k, v in spk2wavs.items() if len(v) >= 2}
    
    logger.info(f"Skipped {skipped_wrong_split} files (Not in Test1).")
    logger.info(f"Loaded {count} Test1 files.")
    logger.info(f"Found {len(valid_speakers)} valid Test1 speakers (with >1 clip).")
    
    return valid_speakers

# ==========================================
# 3. COMPUTE BOUNDS
# ==========================================
def compute_bounds(csv_path, audio_dir, num_pairs=16903):
    model = ECAPA(device=device)
    
    spk2wavs = get_test_set_dict(csv_path, audio_dir)
    
    if len(spk2wavs) < 2:
        logger.error("Not enough Test1 data found. Check your CSV Split column name.")
        return None, None

    speakers = list(spk2wavs.keys())

    # --- Upper Bound (Same Spk) ---
    logger.info("Computing Upper Bound (Test1 GT vs Test1 GT)...")
    upper_scores = []
    # If dataset is small, reduce pairs
    
    for _ in tqdm(range(num_pairs)):
        spk = random.choice(speakers)
        if len(spk2wavs[spk]) < 2: continue # Should be handled, but safe check
        
        f1, f2 = random.sample(spk2wavs[spk], 2)
        sim = model.cosine(model.embed(f1), model.embed(f2))
        if sim is not None: upper_scores.append(sim)

    # --- Lower Bound (Diff Spk) ---
    logger.info("Computing Lower Bound (Random vs Random)...")
    lower_scores = []
    for _ in tqdm(range(num_pairs)):
        s1, s2 = random.sample(speakers, 2)
        f1 = random.choice(spk2wavs[s1])
        f2 = random.choice(spk2wavs[s2])
        sim = model.cosine(model.embed(f1), model.embed(f2))
        if sim is not None: lower_scores.append(sim)

    return np.array(upper_scores), np.array(lower_scores)

if __name__ == "__main__":
    # PATHS
    CSV_PATH = "/sc/home/constantin.auga/New folder/Code/EmoConv-LDM/labels_consensus.csv"
    AUDIO_DIR = "/sc/home/constantin.auga/New folder/Audio/Audio"

    upper, lower = compute_bounds(CSV_PATH, AUDIO_DIR)

    if upper is not None:
        print("\n" + "="*60)
        print("      SPEAKER IDENTITY CONTEXT (Test1 Only)      ")
        print("="*60)
        print(f"Upper Bound (Real Test1 vs Real Test1): {upper.mean():.4f} ± {upper.std():.4f}")
        print(f"Lower Bound (Random vs Random):         {lower.mean():.4f} ± {lower.std():.4f}")
        print("-" * 60)
        print(f"TargetSEC (Your Model):                 0.2882")
        print("="*60)