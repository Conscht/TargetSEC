import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import ast
from torch.nn.utils.rnn import pack_sequence

# ---- paths for cluster ----
#
# MSP-Podcast v1.10 official partitions (labels_consensus.csv Split_Set):
#   Train 63076 | Development 10999 | Test1 16903 | Test2 13289
#
# Training uses Train, validation uses Development, evaluation uses Test1.
# Test1 must never appear in a training loader -- see test_create_data_loader().
_ROOT = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder"

DEFAULT_TENSOR_DIR = f"{_ROOT}/mel_spectograms/Train"
DEFAULT_AUDIO_DIR  = f"{_ROOT}/Audio/Audio"
DEFAULT_META_TRAIN = f"{_ROOT}/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/train.txt"
DEFAULT_EMO_DIR    = f"{_ROOT}/emotion_embeddings"

# Validation split. The Development mel cache is incomplete (8376 of 10999
# utterances have a *_mel.pt); that subset is used as-is.
DEFAULT_VAL_TENSOR_DIR = f"{_ROOT}/mel_spectograms/Development"
DEFAULT_META_VAL       = f"{_ROOT}/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/development.txt"

# Evaluation split -- held out, never used for training or validation.
DEFAULT_TEST_TENSOR_DIR = f"{_ROOT}/mel_spectograms/Test1"
DEFAULT_META_TEST       = f"{_ROOT}/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/test1.txt"

TARGET_N_MELS = 80


class MelSpectrogramDataset(Dataset):
    def __init__(self,
                 tensor_directory: str,
                 embedding_file: str = DEFAULT_META_TRAIN,
                 transform=None,
                 emo_dir: str = None,
                 arousal_csv: str = None):
        self.audio_directory = DEFAULT_AUDIO_DIR
        self.tensor_directory = tensor_directory
        self.transform = transform
        self.emo_dir = emo_dir

        self.file_names = [f for f in os.listdir(tensor_directory)
                           if f.endswith("_mel.pt")]
        self.file_names.sort()

        self.embeddings = self.load_embeddings(embedding_file)
        self.emo_map = self._load_emo_dir(emo_dir) if emo_dir else {}
        self.arousal_map = self._load_arousal_csv(arousal_csv) if arousal_csv else {}
        self.skipped_samples = 0

    def _load_emo_dir(self, emo_dir):
        d = {}
        if not os.path.isdir(emo_dir):
            return d
        for fn in os.listdir(emo_dir):
            if fn.endswith(".pt"):
                d[fn[:-3]] = os.path.join(emo_dir, fn)
        return d

    def _load_arousal_csv(self, csv_path):
        """Load annotated arousal labels from labels_consensus.csv.

        Returns dict {filename_no_ext: scalar} where scalar = (EmoAct-1)/6 ∈ [0,1].
        """
        d = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = os.path.splitext(row["FileName"])[0]
                try:
                    emoact = float(row["EmoAct"])
                    d[fname] = (emoact - 1.0) / 6.0
                except (ValueError, KeyError):
                    pass
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

        if self.arousal_map:
            audio_key_base = os.path.splitext(os.path.basename(audio_file))[0]
            if audio_key_base not in self.arousal_map:
                self.skipped_samples += 1
                return None
            sample["arousal_scalar"] = torch.tensor(
                self.arousal_map[audio_key_base], dtype=torch.float32
            )  # scalar in [0,1]

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

    if "arousal_scalar" in batch[0]:
        out["arousal_scalar"] = torch.stack([b["arousal_scalar"] for b in batch])  # (B,)

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



def _assert_no_test_leak(*tensor_dirs):
    """Guard against training or validating on the evaluation split.

    A stale default silently trained every decoder-stage model on Test1 between
    commits 866ccf07 and this one; this makes that failure loud instead.
    """
    test_dir = os.path.realpath(DEFAULT_TEST_TENSOR_DIR)
    for d in tensor_dirs:
        if d is not None and os.path.realpath(d) == test_dir:
            raise ValueError(
                f"Refusing to build a train/val loader over the evaluation split: {d}. "
                "Test1 is held out; use DEFAULT_TENSOR_DIR (Train) and "
                "DEFAULT_VAL_TENSOR_DIR (Development)."
            )


def _build_loaders(train_dataset, val_dataset, batch_size):
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


def create_dataloaders(batch_size,
                       tensor_dir=DEFAULT_TENSOR_DIR,
                       embedding_file=DEFAULT_META_TRAIN,
                       val_tensor_dir=DEFAULT_VAL_TENSOR_DIR,
                       val_embedding_file=DEFAULT_META_VAL):
    """Train on MSP-Podcast Train, validate on Development.

    Both splits are passed explicitly so that changing one caller can never
    silently change every other caller, as happened with the module constants.
    """
    _assert_no_test_leak(tensor_dir, val_tensor_dir)

    train_dataset = MelSpectrogramDataset(
        tensor_directory=tensor_dir, embedding_file=embedding_file)
    val_dataset = MelSpectrogramDataset(
        tensor_directory=val_tensor_dir, embedding_file=val_embedding_file)

    print(f"[INFO] Train samples: {len(train_dataset)}  (from {tensor_dir})")
    print(f"[INFO] Val   samples: {len(val_dataset)}  (from {val_tensor_dir})")
    print(f"[INFO] Skipped during construction: train={train_dataset.skipped_samples} "
          f"val={val_dataset.skipped_samples}")

    return _build_loaders(train_dataset, val_dataset, batch_size)


DEFAULT_AROUSAL_CSV = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "labels_consensus.csv"
)


def create_dataloaders_with_arousal(batch_size,
                                    arousal_csv=DEFAULT_AROUSAL_CSV,
                                    tensor_dir=DEFAULT_TENSOR_DIR,
                                    embedding_file=DEFAULT_META_TRAIN,
                                    val_tensor_dir=DEFAULT_VAL_TENSOR_DIR,
                                    val_embedding_file=DEFAULT_META_VAL):
    """Dataloaders with annotated arousal labels (from labels_consensus.csv).

    Each batch contains 'arousal_scalar' (B,) with values (EmoAct-1)/6 ∈ [0,1].
    Use this for the faithful HiFiGAN [7] reimplementation so that inference
    with (c-1)/6 stays in-distribution with training.
    """
    _assert_no_test_leak(tensor_dir, val_tensor_dir)

    train_dataset = MelSpectrogramDataset(
        tensor_directory=tensor_dir, embedding_file=embedding_file,
        arousal_csv=arousal_csv)
    val_dataset = MelSpectrogramDataset(
        tensor_directory=val_tensor_dir, embedding_file=val_embedding_file,
        arousal_csv=arousal_csv)

    print(f"[INFO] Train samples (with arousal labels): {len(train_dataset)}  (from {tensor_dir})")
    print(f"[INFO] Val   samples (with arousal labels): {len(val_dataset)}  (from {val_tensor_dir})")
    print(f"[INFO] Skipped: train={train_dataset.skipped_samples} val={val_dataset.skipped_samples}")

    return _build_loaders(train_dataset, val_dataset, batch_size)


def create_dataloaders_with_emotion(batch_size,
                                    emo_dir=DEFAULT_EMO_DIR,
                                    tensor_dir=DEFAULT_TENSOR_DIR,
                                    embedding_file=DEFAULT_META_TRAIN,
                                    val_tensor_dir=DEFAULT_VAL_TENSOR_DIR,
                                    val_embedding_file=DEFAULT_META_VAL):
    """Like create_dataloaders but also returns emotion_emb in each batch."""
    _assert_no_test_leak(tensor_dir, val_tensor_dir)

    train_dataset = MelSpectrogramDataset(
        tensor_directory=tensor_dir, embedding_file=embedding_file, emo_dir=emo_dir)
    val_dataset = MelSpectrogramDataset(
        tensor_directory=val_tensor_dir, embedding_file=val_embedding_file, emo_dir=emo_dir)

    print(f"[INFO] Train samples (with emo): {len(train_dataset)}  (from {tensor_dir})")
    print(f"[INFO] Val   samples (with emo): {len(val_dataset)}  (from {val_tensor_dir})")

    return _build_loaders(train_dataset, val_dataset, batch_size)


def test_create_data_loader(batch_size=1, split="test1"):
    """
    Create a DataLoader for the TEST set.

    Uses the same MelSpectrogramDataset + collate_fn as training,
    but with:
      - test tensor directory (mel_spectrograms)
      - test embedding file (Test*.txt)
      - no random split, just full test set
    """
    if split == "test1":
        tensor_directory = DEFAULT_TEST_TENSOR_DIR
        embedding_file = DEFAULT_META_TEST
    elif split == "dev":
        # Checkpoint selection should not touch the evaluation split. Sweeping
        # checkpoints on Test1 and then reporting Test1 is model selection on
        # the test set; select here, report there.
        tensor_directory = DEFAULT_VAL_TENSOR_DIR
        embedding_file = DEFAULT_META_VAL
    else:
        raise ValueError(f"unknown split {split!r}; expected 'test1' or 'dev'")

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
