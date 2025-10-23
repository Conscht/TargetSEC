import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import ConcordanceCorrCoef
import os
import numpy as np

from src.decoder.decoder_modules import (
    rand_slice_segments, broadcast_embeddings,
    feature_loss, discriminator_loss, generator_loss
)
from processing.preprocessor import get_mel_from_wav, _stft
from src.emotion.emotion_encoder import process_func
from test_audio import calculate_vmos


class EmoSSL(pl.LightningModule):
    def __init__(self, synthesizer, ldm, config):
        super().__init__()
        self.config = config
        self.decoder = synthesizer.decoder
        self.discriminator = synthesizer.discriminator
        self.ldm = ldm

        for p in self.ldm.parameters():
            p.requires_grad = True

        self.audio_time = 2.0
        self.segment_size = 125
        self.dict = synthesizer.dict
        self.stft = _stft.to(self.device)
        self.ccc = ConcordanceCorrCoef(num_outputs=3)

        self.test_outputs = []
        self.y_hat_emo_list = []
        self.y_emo_list = []

        self.automatic_optimization = False

    def slice_audio_mel(self, raw_audio, mel_spec, start_ids):
        audio_start = ((start_ids / 50) * 16000).long()
        audio_end = (audio_start + self.audio_time * 16000).long()

        audio = torch.stack([raw_audio[i, s:e] for i, (s, e) in enumerate(zip(audio_start, audio_end))])
        mel = [mel_spec[i, s//256:e//256] for i, (s, e) in enumerate(zip(audio_start, audio_end))]
        mel = torch.stack([m[:min(m.size(0) for m in mel)] for m in mel])
        return audio, mel

    def generate_style_emb(self, y_audio, speaker_emb):
        with torch.no_grad():
            emo_emb = process_func(y_audio.detach().cpu().numpy(), device=self.device, embeddings=True).unsqueeze(1)
            style_emb = self.ldm(emo_emb, speaker_emb).squeeze(1)
        return style_emb

    def shared_step(self, batch):
        raw_audio, mel_spec = batch['audio'], batch['mel_spectrogram']
        linguistic_emb, lengths = torch.nn.utils.rnn.pad_packed_sequence(batch['hubert'], batch_first=True)
        speaker_emb = batch['speaker_emb']

        x, start_id = rand_slice_segments(linguistic_emb, x_lengths=lengths, segment_size=self.segment_size)
        x = self.dict(x).transpose(1, 2)

        y_audio, y_mel = self.slice_audio_mel(raw_audio, mel_spec, start_id)
        style_emb = self.generate_style_emb(y_audio, speaker_emb)
        x = broadcast_embeddings(x, speaker_emb, style_emb)
        y_hat_audio = self.decoder(x).squeeze(1)

        min_len = min(y_audio.size(-1), y_hat_audio.size(-1))
        y_audio, y_hat_audio = y_audio[..., :min_len], y_hat_audio[..., :min_len]

        y_hat_mel = get_mel_from_wav(y_hat_audio, self.stft)
        y_hat_mel = torch.from_numpy(y_hat_mel).to(y_audio.device)
        y_mel = y_mel.transpose(1, 2)

        if y_hat_mel.size(-1) != y_mel.size(-1):
            min_len = min(y_hat_mel.size(-1), y_mel.size(-1))
            y_hat_mel, y_mel = y_hat_mel[..., :min_len], y_mel[..., :min_len]

        mel_loss = F.l1_loss(y_hat_mel, y_mel)
        return y_audio, y_hat_audio, mel_loss

    def training_step(self, batch, batch_idx):
        y_audio, y_hat_audio, mel_loss = self.shared_step(batch)
        batch_size = y_audio.size(0)

        y_hat_emo = process_func(y_hat_audio.detach().cpu().numpy(), device=self.device, embeddings=False)
        y_emo = process_func(y_audio.detach().cpu().numpy(), device=self.device, embeddings=False)
        ccc_value = self.ccc(y_hat_emo.to(self.ccc.device), y_emo.to(self.ccc.device))
        emo_loss = 1 - ccc_value
        total_loss = 45 * mel_loss + emo_loss.mean()

        self.log('train_mel_loss', mel_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_ccc', emo_loss.mean(), on_epoch=True, prog_bar=True, batch_size=batch_size)

        if self.discriminator:
            opt_g, opt_d = self.optimizers()
            y_audio = y_audio.unsqueeze(1)
            y_hat_audio = y_hat_audio.unsqueeze(1)

            real_pred, fake_pred, fmap_r, fmap_f = self.discriminator(y_audio, y_hat_audio.detach())
            d_loss, _, _ = discriminator_loss(real_pred, fake_pred)
            opt_d.zero_grad(); self.manual_backward(d_loss); opt_d.step()

            _, fake_pred, fmap_r, fmap_f = self.discriminator(y_audio, y_hat_audio)
            loss_fm = feature_loss(fmap_r, fmap_f)
            g_loss = generator_loss(fake_pred)[0] + 2 * loss_fm + total_loss

            opt_g.zero_grad(); self.manual_backward(g_loss); opt_g.step()
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            return total_loss

    def validation_step(self, batch, batch_idx):
        y_audio, y_hat_audio, mel_loss = self.shared_step(batch)
        batch_size = y_audio.size(0)

        if self.current_epoch % 5 == 0:
            y_hat_emo = process_func(y_hat_audio.detach().cpu().numpy(), device=self.device, embeddings=False)
            y_emo = process_func(y_audio.detach().cpu().numpy(), device=self.device, embeddings=False)
            self.y_hat_emo_list.append(y_hat_emo.to(self.ccc.device))
            self.y_emo_list.append(y_emo.to(self.ccc.device))

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            self.logger.experiment.add_audio(f"val_audio_{self.current_epoch}", y_hat_audio[0], self.current_epoch, sample_rate=self.config['data']['sampling_rate'])
            self.logger.experiment.add_audio(f"val_gt_audio_{self.current_epoch}", y_audio[0], self.current_epoch, sample_rate=self.config['data']['sampling_rate'])

        self.log('val_loss', mel_loss * 45, on_step=False, on_epoch=True, batch_size=batch_size)
        return mel_loss

    def on_validation_epoch_end(self):
        if not self.y_hat_emo_list:
            return
        y_hat_all = torch.cat(self.y_hat_emo_list)
        y_all = torch.cat(self.y_emo_list)
        ccc_value = self.ccc(y_hat_all, y_all)
        ccc_loss = 1 - ccc_value
        self.log('val_ccc_arousal', ccc_loss[0].item(), on_epoch=True)
        self.log('val_ccc_valence', ccc_loss[1].item(), on_epoch=True)
        self.log('val_ccc_dominance', ccc_loss[2].item(), on_epoch=True)
        self.log('val_ccc_mean', ccc_loss.mean().item(), on_epoch=True)
        self.y_hat_emo_list.clear()
        self.y_emo_list.clear()

    def test_step(self, batch, batch_idx, emo_embedding, emotion_class):
        """
        Runs inference on a test batch and evaluates the generated audio using VMOS.

        Args:
            batch (dict): Contains 'hubert', 'speaker_emb', 'raw_audio', etc.
            batch_idx (int): Batch index for logging/debugging.
            emo_embedding (Tensor): Precomputed emotion embedding for this emotion class.
            emotion_class (int): Integer label of the emotion class.

        Returns:
            dict: Dictionary with VMOS scores.
        """
        device = next(self.decoder.parameters()).device

        # Extract and move inputs
        linguistic_emb = batch['hubert'].to(device)
        speaker_emb = batch['speaker_emb'].to(device)

        # Pad and process linguistic embeddings
        linguistic_emb, _ = torch.nn.utils.rnn.pad_packed_sequence(linguistic_emb, batch_first=True)
        linguistic_emb = self.dict(linguistic_emb).transpose(1, 2)  # (B, T, C)

        # Ensure correct shape for emo_embedding: (B, 1, D)
        if emo_embedding.ndim == 2:
            emo_embedding = emo_embedding.unsqueeze(1)

        # Generate style embedding from LDM
        with torch.no_grad():
            style_emb = self.ldm(emo_embedding.float(), speaker_emb.float())  # (B, 1, D)

        style_emb = style_emb.squeeze(1)

        # Generate audio with decoder
        embedding = broadcast_embeddings(linguistic_emb, speaker_emb, style_emb)
        print(embedding.shape, "shape final embedd")

        y_hat_audio = self.decoder(embedding).squeeze(1)  # (B, T)

        # Calculate VMOS score for the synthesized audio
        vmos_scores = calculate_vmos(y_hat_audio.cpu(), batch_idx, emotion_class)

        # Log and store results
        self.test_outputs.append({'vmos': vmos_scores})
        self.log(f"test_vmos_class_{emotion_class}", vmos_scores, on_step=False, on_epoch=True, prog_bar=True)

        return {'vmos': vmos_scores}


    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(self.decoder.parameters(), lr=self.config['training']['learning_rate'], betas=(0.8, 0.99), weight_decay=0.01)
        sched_g = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=0.999**(1/8))

        if self.discriminator:
            d_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=self.config['training']['learning_rate'], betas=(0.8, 0.99), weight_decay=0.01)
            sched_d = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=0.999**(1/8))
            return [g_opt, d_opt], [sched_g, sched_d]
        else:
            return [g_opt], [sched_g]

def load_emotion_embeddings(embedding_dir):
    """
    Load emotion embeddings from the Test1 folder.
    The folder contains .npy files where the filename denotes the emotion class.

    Args:
        embedding_dir (str): Path to the Test1 directory of emotion embeddings.

    Returns:
        dict: A dictionary mapping emotion class filenames to their corresponding numpy arrays.
    """
    emotion_embeddings = {}

    # Path to the Test1 folder
    test1_path = os.path.join(embedding_dir, "Test1")
    
    # Load each .npy file inside the Test1 folder (1.npy, 2.npy, etc.)
    for emotion_class in range(1, 8):  # Emotion classes 1 to 7
        embed_path = os.path.join(test1_path, f"{emotion_class}.npy")
        # Load the numpy array for the corresponding emotion class
        emotion_embeddings[emotion_class] = np.load(embed_path)

    return emotion_embeddings
