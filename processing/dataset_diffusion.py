# processing/dataset_diffusion.py
import os, ast, torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import numpy as np

HOP = 256  # 16kHz / 256 hop -> 62.5 fps Mel


# ==== Default paths – adjust to your system ===============================
DEFAULT_TENSOR_DIR = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/mel_spectograms/Train"
DEFAULT_AUDIO_DIR  = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/Audio"
DEFAULT_META_TRAIN = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/train.txt"
DEFAULT_EMO_DIR    = "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/emotion_embeddings"
DEFAULT_META_VAL   = None
# ==========================================================================


class MelSpectrogramDataset(Dataset):
    """
    Returns per item:
      - mel_spectrogram: (T_mel, n_mels)
      - audio:           (T_samps,)
      - mel_original_length: int
      - speaker_emb:     (512,)
      - emotion_emb:     (1024,)  (global; if framewise, we average over time)

    Expects:
      *_mel.pt in tensor_directory
      matching .wav in audio_dir
      matching meta line in embedding_file (with 'spkr_embeds', 'hubert', etc.)
      matching emotion .pt in emo_dir (key = '<basename>.pt', e.g. MSP-PODCAST_0001_0009.pt)
    """
    def __init__(
        self,
        tensor_directory: str,
        embedding_file: str,
        emo_dir: str,
        audio_dir: str,
        expected_mel_len: int = 120,
        expected_audio_len: int = 120 * HOP,
    ):
        self.tensor_directory = tensor_directory
        self.audio_dir = audio_dir
        self.file_names = [f for f in os.listdir(tensor_directory) if f.endswith("_mel.pt")]

        self.embeddings = self._load_meta(embedding_file)
        self.emo_map = self._load_emo_dir(emo_dir)

        self.expected_mel_len = expected_mel_len
        self.expected_audio_len = expected_audio_len
        self.skipped = 0

        # Filter to only files that have both meta and emotion
        def _as_wav(fn): return fn.replace("_mel.pt", ".wav")
        def _base(fn):   return os.path.splitext(_as_wav(fn))[0]

        have_meta = set(self.embeddings.keys())   # "<name>.wav"
        have_emo  = set(self.emo_map.keys())      # "<name>" (no extension)

        valid = [f for f in self.file_names if _as_wav(f) in have_meta and _base(f) in have_emo]

        print(f"[Dataset] kept {len(valid)}/{len(self.file_names)} files "
              f"(meta={len(have_meta)}, emo={len(have_emo)})")
        self.file_names = valid

    def _load_meta(self, embedding_file):
        m = {}
        with open(embedding_file, "r") as f:
            for line in f:
                data = ast.literal_eval(line.strip())
                audio_key = os.path.basename(data["audio"])  # "XXX.wav"
                m[audio_key] = data
        return m

    def _load_emo_dir(self, emo_dir):
        """
        Loads .pt files in emo_dir. Keys are basename without extension.
        E.g. MSP-PODCAST_0001_0009.pt -> key "MSP-PODCAST_0001_0009"
        """
        d = {}
        if emo_dir is None or not os.path.isdir(emo_dir):
            return d
        for fn in os.listdir(emo_dir):
            if not fn.endswith(".pt"):
                continue
            key_no_ext = fn[:-3]
            d[key_no_ext] = torch.load(os.path.join(emo_dir, fn), map_location="cpu")
        return d

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        mel_path = os.path.join(self.tensor_directory, self.file_names[idx])
        mel = torch.load(mel_path, map_location="cpu")

        audio_key = self.file_names[idx].replace("_mel.pt", ".wav")
        audio_key_base = os.path.splitext(audio_key)[0]

        if audio_key not in self.embeddings or audio_key_base not in self.emo_map:
            self.skipped += 1
            return None

        # speaker embedding from meta
        speaker_emb = torch.tensor(
            self.embeddings[audio_key]["spkr_embeds"],
            dtype=torch.float32
        )

        # audio
        wav_path = os.path.join(self.audio_dir, audio_key)
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0).float()  # mono

        # robust mel -> torch.float (T, n_mels)
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        elif not torch.is_tensor(mel):
            mel = torch.tensor(mel)
        mel = mel.float().contiguous()

        if mel.dim() == 3 and mel.size(0) == 1:
            mel = mel.squeeze(0)
        if mel.size(0) < mel.size(1):  # (n_mels, T) -> (T, n_mels)
            mel = mel.transpose(0, 1).contiguous()

        # length check
        if mel.size(0) < self.expected_mel_len or wav.size(0) < self.expected_audio_len:
            self.skipped += 1
            return None

        # emotion embedding from emo_map
        emotion_emb = self.emo_map[audio_key_base]
        if isinstance(emotion_emb, np.ndarray):
            emotion_emb = torch.from_numpy(emotion_emb)
        elif not torch.is_tensor(emotion_emb):
            emotion_emb = torch.tensor(emotion_emb)
        emotion_emb = emotion_emb.float()

        # If framewise, average to global (T_emo, D) -> (D,)
        if emotion_emb.dim() > 1:
            emotion_emb = emotion_emb.view(-1, emotion_emb.size(-1)).mean(dim=0)

        return {
            "mel_spectrogram": mel,             # (T_mel, n_mels)
            "audio": wav,                       # (T_samps,)
            "mel_original_length": mel.size(0),
            "speaker_emb": speaker_emb,         # (512,)
            "emotion_emb": emotion_emb,         # (1024,)
        }


def collate_fn(batch):
    """
    Pads & stacks to tensors:
      mel_spectrogram:    (B, Tm, n_mels)
      audio:              (B, Ts)
      mel_original_lengths: (B,)
      speaker_emb:        (B, 512)
      emotion_emb:        (B, 1024)
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    Tm = max(b["mel_spectrogram"].size(0) for b in batch)
    Ts = max(b["audio"].size(0) for b in batch)
    n_mels = batch[0]["mel_spectrogram"].size(1)

    mel = torch.stack([
        torch.cat(
            [b["mel_spectrogram"],
             torch.zeros(Tm - b["mel_spectrogram"].size(0), n_mels)],
            dim=0
        )
        for b in batch
    ])  # (B, Tm, n_mels)

    audio = torch.stack([
        torch.cat(
            [b["audio"],
             torch.zeros(Ts - b["audio"].size(0))],
            dim=0
        )
        for b in batch
    ])  # (B, Ts)

    mel_len = torch.tensor(
        [b["mel_original_length"] for b in batch],
        dtype=torch.long
    )

    spk = torch.stack([b["speaker_emb"] for b in batch])   # (B, 512)
    emo = torch.stack([b["emotion_emb"] for b in batch])   # (B, 1024)

    return {
        "mel_spectrogram": mel,
        "audio": audio,
        "mel_original_lengths": mel_len,
        "speaker_emb": spk,
        "emotion_emb": emo,
    }


def _make_loader(tensor_dir, meta_file, emo_dir, audio_dir, batch_size, shuffle):
    ds = MelSpectrogramDataset(
        tensor_directory=tensor_dir,
        embedding_file=meta_file,
        emo_dir=emo_dir,
        audio_dir=audio_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader


def create_dataloaders(
    batch_size: int,
    train_tensor_dir: str = DEFAULT_TENSOR_DIR,
    val_tensor_dir: str   = None,
    audio_dir: str        = DEFAULT_AUDIO_DIR,
    train_meta: str       = DEFAULT_META_TRAIN,
    val_meta: str         = DEFAULT_META_VAL,
    emo_dir: str          = DEFAULT_EMO_DIR,
):
    """
    Returns train_loader, val_loader.
    If val_tensor_dir is None, split train folder 80/20.
    """
    if val_tensor_dir is None:
        full_ds = MelSpectrogramDataset(
            tensor_directory=train_tensor_dir,
            embedding_file=train_meta,
            emo_dir=emo_dir,
            audio_dir=audio_dir,
        )
        n = len(full_ds)
        trn = int(0.8 * n)
        val = n - trn
        g = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [trn, val], generator=g)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, persistent_workers=True, pin_memory=True,
            collate_fn=collate_fn, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=max(1, batch_size // 2), shuffle=False,
            num_workers=4, persistent_workers=True, pin_memory=True,
            collate_fn=collate_fn, drop_last=False
        )
        return train_loader, val_loader

    # Separate train/val dirs
    train_loader = _make_loader(train_tensor_dir, train_meta, emo_dir, audio_dir, batch_size, shuffle=True)
    val_loader   = _make_loader(val_tensor_dir,   val_meta,   emo_dir, audio_dir, max(1, batch_size // 2), shuffle=False)
    return train_loader, val_loader
