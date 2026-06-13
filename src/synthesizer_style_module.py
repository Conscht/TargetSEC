from src.decoder.decoder_modules import rand_slice_segments, broadcast_embeddings, feature_loss, discriminator_loss, generator_loss
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch

from torch import nn
from processing.preprocessor import get_mel_from_wav, _stft
import torch.nn.functional as F
import pytorch_lightning as pl
from test_audio import calculate_vmos
from src.emotion.emotion_encoder import process_func
from torchmetrics.regression import ConcordanceCorrCoef


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SynthesizerLightningModule(pl.LightningModule):
    def __init__(self, style_encoder, decoder, config, discriminator=None):
        super().__init__()
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.config = config
        self.automatic_optimization = False

        self.audio_time = 2.5      # seconds
        self.segment_size = 125    # HuBERT frames for 2.5s @ 50 Hz
        self.dict = nn.Embedding(100, 128)
        self.stft = _stft.to(device)

        self.test_outputs = []
        self.y_hat_emo_list = []
        self.y_emo_list = []
        self.ccc = ConcordanceCorrCoef(num_outputs=3)

    def forward(self, x):
        style_emb = self.style_encoder(x)
        return self.decoder(style_emb)

    def slice_audio_and_mel(self, raw_audio, mel_spec, start_ids):
        """
        raw_audio: (B, T_audio_max)
        mel_spec:  (B, T_mel_max, n_mels)
        start_ids: (B,) start index in HuBERT frames (50 Hz)

        Returns:
          audio_segments: (B, win_audio)          # 2.5s = 40000 samples
          mel_segments:   (B, win_mel, n_mels)    # ≈156 frames
        """
        sr = 16000
        hubert_rate = 50.0
        hop = 256

        win_audio = int(self.audio_time * sr)  # 2.5s -> 40000
        win_mel = win_audio // hop            # ~156

        B = raw_audio.size(0)
        T_audio = raw_audio.size(1)
        T_mel = mel_spec.size(1)
        n_mels = mel_spec.size(2)

        audio_segments = []
        mel_segments = []

        for i in range(B):
            # --- map HuBERT start to audio samples ---
            s_h = start_ids[i].item()  # HuBERT index
            start_sample = int(s_h / hubert_rate * sr)  # samples

            end_sample = start_sample + win_audio

            # clamp to valid range (within padded audio)
            if end_sample > T_audio:
                end_sample = T_audio
                start_sample = max(0, end_sample - win_audio)

            audio_seg = raw_audio[i, start_sample:end_sample]  # (<= win_audio,)

            # pad to exactly win_audio
            if audio_seg.size(0) < win_audio:
                pad_len = win_audio - audio_seg.size(0)
                audio_seg = torch.cat(
                    [audio_seg, audio_seg.new_zeros(pad_len)],
                    dim=0
                )

            audio_segments.append(audio_seg.unsqueeze(0))

            # --- same region in mel frames ---
            start_mel = start_sample // hop
            end_mel = start_mel + win_mel

            if end_mel > T_mel:
                end_mel = T_mel
                start_mel = max(0, end_mel - win_mel)

            mel_seg = mel_spec[i, start_mel:end_mel, :]  # (<= win_mel, n_mels)

            if mel_seg.size(0) < win_mel:
                pad_len = win_mel - mel_seg.size(0)
                mel_seg = torch.cat(
                    [mel_seg, mel_seg.new_zeros(pad_len, n_mels)],
                    dim=0
                )

            mel_segments.append(mel_seg.unsqueeze(0))

        audio_segments = torch.cat(audio_segments, dim=0)  # (B, win_audio)
        mel_segments = torch.cat(mel_segments, dim=0)      # (B, win_mel, n_mels)

        return audio_segments, mel_segments

    def shared_step(self, batch):
        raw_audio, mel_spec = batch['audio'], batch['mel_spectrogram']
        linguistic_emb, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            batch['hubert'], batch_first=True
        )
        speaker_emb = batch['speaker_emb']

        # 1) random 2.5s window in HuBERT space (always 125 frames, padded if needed)
        x_ids, start_id = rand_slice_segments(
            linguistic_emb, x_lengths=lengths, segment_size=self.segment_size
        )  # x_ids: (B, 125)

        # 2) embed HuBERT ids
        x = self.dict(x_ids).transpose(1, 2)  # (B, C, 125)

        # 3) get matching 2.5s audio+mel window
        y_audio, mel_segment = self.slice_audio_and_mel(raw_audio, mel_spec, start_id)
        # mel_segment: (B, T_mel_win, n_mels)

        # 4) style encoder on mel
        with torch.no_grad():
            style_emb = self.style_encoder(mel_segment)

        x = broadcast_embeddings(x, speaker_emb, style_emb)
        y_hat_audio = self.decoder(x).squeeze(1)  # (B, win_audio)

        # 5) mel loss
        y_hat_mel = get_mel_from_wav(y_hat_audio, self.stft)
        y_hat_mel = torch.from_numpy(y_hat_mel).to(y_audio.device)  # (B, n_mels, T_mel)
        y_mel = mel_segment.transpose(1, 2)                          # (B, n_mels, T_mel)

        # trim to common time length (should already match, but safe)
        if y_hat_mel.shape[-1] != y_mel.shape[-1]:
            min_len = min(y_hat_mel.shape[-1], y_mel.shape[-1])
            y_hat_mel, y_mel = y_hat_mel[..., :min_len], y_mel[..., :min_len]

        mel_loss = F.l1_loss(y_hat_mel, y_mel)

        return y_audio, y_hat_audio, mel_loss

    def training_step(self, batch, batch_idx):
        if batch is None:
            return
        y_audio, y_hat_audio, mel_loss = self.shared_step(batch)
        batch_size = y_audio.size(0)

        device = y_audio.device
        y_hat_emo = process_func(y_hat_audio.detach().cpu().numpy(),
                                 device=device, embeddings=False)
        y_emo = process_func(y_audio.detach().cpu().numpy(),
                             device=device, embeddings=False)

        y_hat_emo = y_hat_emo.float().to(self.ccc.device)
        y_emo = y_emo.float().to(self.ccc.device)

        ccc_value = self.ccc(y_hat_emo, y_emo)
        emo_loss = 1 - ccc_value

        total_loss = 45 * mel_loss + emo_loss.mean()
        self.log('train_mel_loss', mel_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_ccc', emo_loss.mean(), on_epoch=True, prog_bar=True, batch_size=batch_size)

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            self.logger.experiment.add_audio(
                f"train_audio_{self.current_epoch}", y_hat_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )
            self.logger.experiment.add_audio(
                f"train_gt_audio_{self.current_epoch}", y_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )

        if self.discriminator:
            optimizer_g, optimizer_d = self.optimizers()

            y_audio_d = y_audio.unsqueeze(1)
            y_hat_audio_d = y_hat_audio.unsqueeze(1)

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(
                y_audio_d, y_hat_audio_d.detach()
            )
            d_loss, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(
                y_audio_d, y_hat_audio_d
            )
            loss_fm = feature_loss(fmap_r, fmap_g)
            g_adv_loss, _ = generator_loss(y_d_hat_g)
            g_loss = g_adv_loss + 2 * loss_fm + total_loss

            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()

            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            return total_loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return

        y_audio, y_hat_audio, mel_loss = self.shared_step(batch)
        batch_size = y_audio.size(0)

        if self.current_epoch % 5 == 0:
            device = y_audio.device
            y_hat_emo = process_func(y_hat_audio.cpu().numpy(),
                                     device=device, embeddings=False)
            y_emo = process_func(y_audio.cpu().numpy(),
                                 device=device, embeddings=False)

            self.y_hat_emo_list.append(y_hat_emo.to(self.ccc.device))
            self.y_emo_list.append(y_emo.to(self.ccc.device))

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            self.logger.experiment.add_audio(
                f"val_audio_{self.current_epoch}", y_hat_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )
            self.logger.experiment.add_audio(
                f"val_gt_audio_{self.current_epoch}", y_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )

        self.log('val_loss', mel_loss * 45, on_step=False, on_epoch=True, batch_size=batch_size)
        return mel_loss

    def on_validation_epoch_end(self):
        if not self.y_hat_emo_list:
            return
        y_hat_emo_all = torch.cat(self.y_hat_emo_list, dim=0)
        y_emo_all = torch.cat(self.y_emo_list, dim=0)
        ccc_value = self.ccc(y_hat_emo_all, y_emo_all)
        ccc_loss = 1 - ccc_value

        self.log('val_ccc_arousal',   ccc_loss[0].item(), on_epoch=True)
        self.log('val_ccc_valence',   ccc_loss[1].item(), on_epoch=True)
        self.log('val_ccc_dominance', ccc_loss[2].item(), on_epoch=True)
        self.log('val_ccc_mean',      ccc_loss.mean().item(), on_epoch=True)

        self.y_hat_emo_list.clear()
        self.y_emo_list.clear()

    def test_step(self, batch, batch_idx):
        mel_spec = batch['mel_spectrogram']
        linguistic_emb, _ = torch.nn.utils.rnn.pad_packed_sequence(
            batch['hubert'], batch_first=True
        )
        speaker_emb = batch['speaker_emb']

        x = self.dict(linguistic_emb).transpose(1, 2)
        with torch.no_grad():
            style = self.style_encoder(mel_spec)
        x = broadcast_embeddings(x, speaker_emb, style)
        output = self.decoder(x)

        vmos = calculate_vmos(output.cpu(), batch_idx)
        self.test_outputs.append({'loss': vmos})
        return {'loss': vmos}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        print(f"Test Completed - Average Loss: {avg_loss.item()}")
        self.test_outputs.clear()

    def configure_optimizers(self):
        g_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(0.8, 0.99),
            weight_decay=0.01
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            g_optimizer, gamma=0.999**(1/8)
        )

        if self.discriminator:
            d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config['training']['learning_rate'],
                betas=(0.8, 0.99),
                weight_decay=0.01
            )

            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                d_optimizer, gamma=0.999**(1/8)
            )
            return [g_optimizer, d_optimizer], [scheduler_g, scheduler_d]
        else:
            return [g_optimizer], [scheduler_g]
