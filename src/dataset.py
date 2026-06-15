import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import numpy as np
import ast
from torch.nn.utils.rnn import pack_sequence

# ---- paths for cluster ----
DEFAULT_TENSOR_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Test1"
DEFAULT_AUDIO_DIR  = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/Audio"
DEFAULT_META_TRAIN = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"
DEFAULT_EMO_DIR    = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/emotion_embeddings"

TARGET_N_MELS = 80


class MelSpectrogramDataset(Dataset):
    def __init__(self,
                 tensor_directory: str,
                 embedding_file: str = DEFAULT_META_TRAIN,
                 transform=None,
                 emo_dir: str = None):
        self.audio_directory = DEFAULT_AUDIO_DIR
        self.tensor_directory = tensor_directory
        self.transform = transform
        self.emo_dir = emo_dir

        self.file_names = [f for f in os.listdir(tensor_directory)
                           if f.endswith("_mel.pt")]
        self.file_names.sort()

        self.embeddings = self.load_embeddings(embedding_file)
        self.emo_map = self._load_emo_dir(emo_dir) if emo_dir else {}
        self.skipped_samples = 0

    def _load_emo_dir(self, emo_dir):
        d = {}
        if not os.path.isdir(emo_dir):
            return d
        for fn in os.listdir(emo_dir):
            if fn.endswith(".pt"):
                d[fn[:-3]] = os.path.join(emo_dir, fn)
        return d

    def load_embeddings(self, embedding_file):
        embeddings = {}
        with open(embedding_file, "r") as f:
            for line in f:
                data = ast.literal_eval(line.strip())
                audio_key = os.path.basename(data["audio"])
                embeddings[audio_key] = data
        return embeddings

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # ----- mel -----
        mel_file = os.path.join(self.tensor_directory,
                                self.file_names[index])
        mel = torch.load(mel_file)

        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)

        # (1, 80, T) -> (80, T)
        if mel.dim() == 3 and mel.size(0) == 1:
            mel = mel.squeeze(0)

        if mel.dim() != 2:
            print(f"[WARN] Unexpected mel dim {mel.shape} in {mel_file}, skipping.")
            self.skipped_samples += 1
            return None

        # At this point, from your scan, mel is (80, T) for all files.
        # We want final shape (T, 80) for the model, same as old code.
        if mel.size(0) == TARGET_N_MELS:
            # (80, T) -> (T, 80)
            mel = mel.transpose(0, 1).contiguous()
        elif mel.size(1) == TARGET_N_MELS:
            # already (T, 80)
            pass
        else:
            print(f"[WARN] n_mels != {TARGET_N_MELS} for {mel_file}: {mel.shape}, skipping.")
            self.skipped_samples += 1
            return None

        mel_spectrogram = mel  # (T, 80)

        # ----- audio -----
        audio_file = os.path.join(
            self.audio_directory,
            self.file_names[index].replace("_mel.pt", ".wav")
        )
        audio = self.load_audio(audio_file)  # 1D: (T_audio,)

        # ----- embeddings -----
        audio_key = os.path.basename(audio_file)
        if audio_key not in self.embeddings:
            print(f"[WARN] Embedding for {audio_key} not found, skipping.")
            self.skipped_samples += 1
            return None

        hubert_embedding = torch.tensor(
            [int(x) for x in self.embeddings[audio_key]["hubert"].split()],
            dtype=torch.long,
        )
        speaker_embedding = torch.tensor(
            self.embeddings[audio_key]["spkr_embeds"],
            dtype=torch.float32,
        )

        sample = {
            "mel_spectrogram": mel_spectrogram,  # (T_mel, 80)
            "audio": audio,                      # (T_audio,)
            "hubert": hubert_embedding,          # (L_hubert,)
            "speaker_emb": speaker_embedding,    # (D_spkr,)
        }

        if self.emo_map:
            audio_key_base = os.path.splitext(os.path.basename(audio_file))[0]
            if audio_key_base not in self.emo_map:
                self.skipped_samples += 1
                return None
            emo = torch.load(self.emo_map[audio_key_base], map_location="cpu").float()
            if emo.dim() > 1:
                emo = emo.view(-1, emo.size(-1)).mean(dim=0)
            sample["emotion_emb"] = emo  # (1024,)

        return sample

    def load_audio(self, file_path):
        try:
            audio, _ = torchaudio.load(file_path)
            return audio.squeeze(0)
        except Exception as e:
            print(f"Failed to load file {file_path}: {str(e)}")
            raise


def collate_fn(batch):
    # drop Nones
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        print("[WARNING] All items in batch were skipped.")
        return None

    mels = [b["mel_spectrogram"] for b in batch]  # (T_mel, 80)
    audios = [b["audio"] for b in batch]          # (T_audio,)
    huberts = [b["hubert"] for b in batch]        # 1D
    speaker_embs = [b["speaker_emb"] for b in batch]

    # pad audio (time dim)
    max_len_audio = max(a.size(0) for a in audios)
    padded_audios = [
        torch.cat([a, a.new_zeros(max_len_audio - a.size(0))], dim=0)
        if a.size(0) < max_len_audio else a
        for a in audios
    ]

    # pad mel (time dim)
    max_len_mel = max(m.size(0) for m in mels)
    padded_mels = [
        torch.cat(
            [m, m.new_zeros(max_len_mel - m.size(0), m.size(1))],
            dim=0
        ) if m.size(0) < max_len_mel else m
        for m in mels
    ]

    # pack HuBERT
    packed_huberts = pack_sequence(huberts, enforce_sorted=False)

    # audio attention mask (1 = real, 0 = padded)
    audio_lengths = [a.size(0) for a in audios]
    audio_attention_mask = torch.stack([
        torch.cat([
            torch.ones(L),
            torch.zeros(max_len_audio - L)
        ])
        for L in audio_lengths
    ])

    out = {
        "mel_spectrogram": torch.stack(padded_mels),   # (B, T_mel_max, 80)
        "audio": torch.stack(padded_audios),           # (B, T_audio_max)
        "hubert": packed_huberts,
        "speaker_emb": torch.stack(speaker_embs),
        "mel_original_lengths": audio_lengths,
        "audio_attention_mask": audio_attention_mask,
    }

    if "emotion_emb" in batch[0]:
        out["emotion_emb"] = torch.stack([b["emotion_emb"] for b in batch])  # (B, 1024)

    return out

def collate_fn_stats(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    mels = [b["mel_spectrogram"] for b in batch]   # each is (T, 80), unpadded
    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)  # ✅ true mel lengths

    max_len_mel = int(mel_lengths.max().item())
    n_mels = mels[0].size(1)

    padded_mels = [
        torch.cat([m, m.new_zeros(max_len_mel - m.size(0), n_mels)], dim=0)
        if m.size(0) < max_len_mel else m
        for m in mels
    ]

    return {
        "mel_spectrogram": torch.stack(padded_mels),   # (B, Tm, 80)
        "mel_lengths": mel_lengths,                    # (B,)
    }



def create_dataloaders(batch_size, val_split=0.2):
    tensor_directory = DEFAULT_TENSOR_DIR

    full_dataset = MelSpectrogramDataset(tensor_directory=tensor_directory)
    print(f"[INFO] Total samples found: {len(full_dataset)}")
    print(f"[INFO] Skipped samples during dataset construction: {full_dataset.skipped_samples}")

    torch.manual_seed(42)
    train_size = int((1.0 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def create_dataloaders_with_emotion(batch_size, val_split=0.2, emo_dir=DEFAULT_EMO_DIR):
    """Like create_dataloaders but also returns emotion_emb in each batch."""
    full_dataset = MelSpectrogramDataset(
        tensor_directory=DEFAULT_TENSOR_DIR,
        emo_dir=emo_dir,
    )
    print(f"[INFO] Total samples (with emo): {len(full_dataset)}")

    torch.manual_seed(42)
    train_size = int((1.0 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def test_create_data_loader(batch_size=1):
    """
    Create a DataLoader for the TEST set.

    Uses the same MelSpectrogramDataset + collate_fn as training,
    but with:
      - test tensor directory (mel_spectrograms)
      - test embedding file (Test*.txt)
      - no random split, just full test set
    """
    tensor_directory = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Test1"
    embedding_file = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"


    full_dataset = MelSpectrogramDataset(
        tensor_directory=tensor_directory,
        embedding_file=embedding_file,
    )

    print(f"[INFO] Test samples found: {len(full_dataset)}")
    print(f"[INFO] Skipped samples during test dataset construction: {full_dataset.skipped_samples}")

    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        collate_fn=collate_fn,
    )

    return test_loader
