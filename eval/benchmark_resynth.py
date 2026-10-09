"""
Resynthesis benchmark: fine-tuned synthesizer only, NO LDM.

For each test utterance:
  mel → fine-tuned style encoder (center 2.5s window) → style vector
  (HuBERT + speaker + style) → decoder → audio

Saves WAVs to eval_outputs/resynth_epoch48/wav/ and prints WVMOS.
No LDM involved → no style-space mismatch. This measures synthesizer
+ style encoder quality in isolation.
"""
import os

import torch
import torchaudio

from StyleSpeech.models.StyleSpeech import MelStyleEncoder
from config.stylespeech_model_config import style_config
from src.synthesizer_style_module import SynthesizerLightningModule
from src.decoder.decoder import Generator, MultiPeriodDiscriminator
from src.dataset import test_create_data_loader
from src.decoder.decoder_modules import broadcast_embeddings

CHECKPOINT_SYNTH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/Code/EmoConv-LDM/"
    "checkpoints_synthesizer_finetune/"
    "synthesizer_finetune-06-13_18-22-18-epoch=48-val_loss=16.24.ckpt"
)
PRETRAINED_STYLE_PATH = (
    "/sc/projects/sci-demelo/mpws2025gd1/constantin/New folder/"
    "Audio/MSP-Podcast-1.10/pre-trained_models/pre-trained_style"
)
SAVE_ROOT = "eval_outputs/resynth_epoch48"

SEGMENT_FRAMES = 156   # 2.5 s at 16 kHz / 256 hop = 156 mel frames

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
        "mel_fmax": 8000.0,
    },
    "training": {"learning_rate": 5e-5, "batch_size": 8},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_center_style(mel_spec, style_encoder, seg_frames=SEGMENT_FRAMES):
    """Extract style from center window of mel spectrogram (same as training)."""
    T = mel_spec.shape[1]
    half = seg_frames // 2
    center = T // 2
    s = max(0, center - half)
    e = min(T, s + seg_frames)
    mel_win = mel_spec[:, s:e, :]
    return style_encoder(mel_win)  # (1, 128)


if __name__ == "__main__":
    os.makedirs(os.path.join(SAVE_ROOT, "wav"), exist_ok=True)

    style_encoder = MelStyleEncoder(style_config)
    style_encoder.load_state_dict(torch.load(PRETRAINED_STYLE_PATH, map_location="cpu"))

    gen = Generator(config_synth)
    discrim = MultiPeriodDiscriminator()

    print(f"Loading synthesizer from: {CHECKPOINT_SYNTH}")
    synthesizer = SynthesizerLightningModule.load_from_checkpoint(
        CHECKPOINT_SYNTH,
        style_encoder=style_encoder,
        decoder=gen,
        discriminator=discrim,
        config=config_synth,
    ).to(device).eval()

    for p in synthesizer.parameters():
        p.requires_grad = False

    decoder    = synthesizer.decoder.to(device).eval()
    dict_proj  = synthesizer.dict.to(device).eval()
    enc        = synthesizer.style_encoder.to(device).eval()

    sr = config_synth["data"]["sampling_rate"]
    test_loader = test_create_data_loader(batch_size=1)

    for batch_idx, batch in enumerate(test_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        linguistic, _ = torch.nn.utils.rnn.pad_packed_sequence(
            batch["hubert"], batch_first=True
        )
        linguistic = dict_proj(linguistic.to(device)).transpose(1, 2)
        speaker    = batch["speaker_emb"]          # (1, 512)
        mel_spec   = batch["mel_spectrogram"]      # (1, T, 80)

        with torch.inference_mode():
            style = extract_center_style(mel_spec, enc)   # (1, 128)
            emb   = broadcast_embeddings(linguistic, speaker, style)
            y_hat = decoder(emb).squeeze(1).clamp(-1.0, 1.0)

        out_path = os.path.join(SAVE_ROOT, "wav", f"utt_{batch_idx:06d}.wav")
        torchaudio.save(out_path, y_hat[0].detach().cpu().unsqueeze(0), sample_rate=sr)

        if batch_idx % 500 == 0 and batch_idx > 0:
            print(f"[{batch_idx}] utterances processed")

    print(f"\nDone. WAVs saved to {SAVE_ROOT}/wav/")
    print("Run wvmos_calc_resynth.py to compute WVMOS.")
