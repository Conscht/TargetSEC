from src.decoder.decoder_modules import rand_slice_segments, broadcast_embeddings, feature_loss, discriminator_loss, generator_loss
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch

from torch import nn
from processing.mel_differentiable import mel_spectrogram_torch
import torch.nn.functional as F
import pytorch_lightning as pl
from test_audio import calculate_vmos
from src.emotion.emotion_encoder import DifferentiableSER, concordance_cc
from torchmetrics.regression import ConcordanceCorrCoef

# Same weights as the HiFiGAN baseline so the two systems share an objective
# scale. Previously lambda_fm was effectively 4: decoder_modules.feature_loss
# already returns loss*2, and the call site multiplied by 2 again.
LAMBDA_FM = 2.0
LAMBDA_RECON = 45.0
LAMBDA_EMO = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SynthesizerLightningModule(pl.LightningModule):
    def __init__(self, style_encoder, decoder, config, discriminator=None,
                 freeze_style_encoder=False, ser_loss_mode="differentiable"):
        """Two-stage recipe.

        Stage 1 (train_synthesizer.py): `freeze_style_encoder=True`. The
        LibriTTS-pretrained MelStyleEncoder is held fixed and only the decoder,
        unit embedding and discriminator train against it.

        Stage 2 (train_synthesizer_finetune.py): `freeze_style_encoder=False`.
        The style encoder joins the optimizer at 0.1x the decoder LR.

        This was the original design -- before commit 1eaea02a the style encoder
        was simply omitted from the optimizer in configure_optimizers, which
        froze it by exclusion. That commit added it unconditionally, collapsing
        the two stages into one. The flag restores the distinction.

        It matters more now than it did then: with the mel loss finally
        differentiable, an unfrozen style encoder receives 45x a live L1 rather
        than only adversarial/feature-matching gradients, so it has a strong
        incentive to encode utterance-specific acoustic detail into its 128
        dims. That lowers reconstruction loss but makes the latent harder for
        the LDM to predict from emotion + speaker alone. Watch `style_emb_std`.
        """
        super().__init__()
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.config = config
        self.freeze_style_encoder = freeze_style_encoder
        # "detached" logs the CCC term without a gradient. Use it to keep the
        # decoder from being optimised through the same audeering model that
        # scores the benchmark -- the baseline hits forced targets at L_abs
        # 0.052 that way, versus 0.098 for the scorer's own agreement with human
        # labels on real speech, i.e. below the metric's noise floor.
        assert ser_loss_mode in ("differentiable", "detached")
        self.ser_loss_mode = ser_loss_mode
        self.automatic_optimization = False

        if freeze_style_encoder:
            self.style_encoder.eval()
            for p in self.style_encoder.parameters():
                p.requires_grad_(False)

        self.audio_time = 2.5      # seconds
        self.segment_size = 125    # HuBERT frames for 2.5s @ 50 Hz
        self.dict = nn.Embedding(100, 128)

        # Frozen SER for the emotion loss. Held in a plain list so nn.Module
        # never registers it: it must stay out of the state_dict (~1.2 GB per
        # checkpoint) and must never be flipped out of eval by .train().
        self._ser_holder = [DifferentiableSER()]

        self.test_outputs = []
        self.y_hat_emo_list = []
        self.y_emo_list = []
        # Validation-only metric, kept separate from the (stateless) training
        # loss: a torchmetrics Metric accumulates until reset, so one shared
        # instance silently pools train and val into the logged value.
        self.val_ccc = ConcordanceCorrCoef(num_outputs=3)

    @property
    def ser(self):
        """Frozen SER, moved to the module's device on first use."""
        m = self._ser_holder[0]
        if next(m.parameters()).device != self.device:
            m.to(self.device)
        return m

    def _has_audio_logger(self):
        logger = self.logger
        return (logger is not None
                and getattr(logger, "experiment", None) is not None
                and hasattr(logger.experiment, "add_audio"))

    def train(self, mode: bool = True):
        """Keep a frozen style encoder in eval mode.

        Lightning calls .train() on the whole module at every epoch, which would
        otherwise re-enable dropout/BN updates inside the style encoder even
        though its weights are excluded from the optimizer.
        """
        super().train(mode)
        if self.freeze_style_encoder:
            self.style_encoder.eval()
        return self

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

            # pad to exactly win_audio using reflection to avoid hard silence at boundary
            if audio_seg.size(0) < win_audio:
                pad_len = win_audio - audio_seg.size(0)
                audio_seg = torch.nn.functional.pad(
                    audio_seg.unsqueeze(0).unsqueeze(0), (0, pad_len), mode='reflect'
                ).squeeze(0).squeeze(0)

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
                mel_seg = torch.nn.functional.pad(
                    mel_seg.unsqueeze(0).permute(0, 2, 1), (0, pad_len), mode='reflect'
                ).permute(0, 2, 1).squeeze(0)

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

        # 4) style encoder on mel — fine-tuned end-to-end at lower LR
        style_emb = self.style_encoder(mel_segment)
        self.log('style_emb_std', style_emb.detach().std(), prog_bar=False)

        x = broadcast_embeddings(x, speaker_emb, style_emb)
        y_hat_audio = self.decoder(x).squeeze(1)  # (B, win_audio)

        # 5) mel loss -- both sides recomputed from the waveforms.
        #
        # The old path was get_mel_from_wav(...) -> torch.from_numpy(...), which
        # returns a leaf with requires_grad=False, so 45 * mel_loss was a
        # constant and L_recon trained nothing. It also compared against
        # mel_segment, sliced at start_sample // 256 while the audio was sliced
        # at s_h * 320, leaving target and prediction up to one frame apart.
        #
        # The style encoder above still consumes the cached mel_segment, so its
        # input distribution -- and compute_style_stats_finetune.py -- are
        # unchanged.
        y_hat_mel = mel_spectrogram_torch(y_hat_audio)   # (B, n_mels, T_mel)
        y_mel = mel_spectrogram_torch(y_audio)           # (B, n_mels, T_mel)

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

        # Emotion loss: 1 - CCC over (arousal, valence, dominance).
        #
        # Now differentiable. The old code ran process_func on
        # y_hat_audio.detach().cpu().numpy() under torch.no_grad(), so emo_loss
        # was a constant and this term trained nothing.
        #
        # Target is kept as SER(y_real) -- deliberate, and the opposite of the
        # HiFiGAN baseline. TargetSEC's style vector is derived from the input
        # utterance's own mel, so this is a resynthesis-consistency term:
        # "reproduce the emotion that was there". The baseline instead has to
        # honour an externally supplied target, so its L_SER is scored against
        # the annotated label. Swapping this to labels_consensus.csv values
        # (EmoAct/EmoVal/EmoDom) is a one-line change if you want the two
        # systems to share a target.
        #
        # CCC is a batch statistic: meaningless at batch size 1, noisy below ~16.
        if self.ser_loss_mode == "detached":
            with torch.no_grad():
                y_hat_emo = self.ser(y_hat_audio)
                y_emo = self.ser(y_audio)
        else:
            y_hat_emo = self.ser(y_hat_audio)                  # (B, 3), grad flows
            with torch.no_grad():
                y_emo = self.ser(y_audio)                      # (B, 3), constant target

        emo_loss = torch.stack([
            1.0 - concordance_cc(y_emo[:, d], y_hat_emo[:, d]) for d in range(3)
        ]).mean()

        total_loss = LAMBDA_RECON * mel_loss
        if self.ser_loss_mode != "detached":
            total_loss = total_loss + LAMBDA_EMO * emo_loss
        self.log('train_mel_loss', mel_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_ccc', emo_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

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
            # feature_loss already returns loss*2, so the old `2 * loss_fm`
            # applied lambda_fm = 4 rather than the intended 2.
            loss_fm = feature_loss(fmap_r, fmap_g) / 2.0
            g_adv_loss, _ = generator_loss(y_d_hat_g)
            g_loss = g_adv_loss + LAMBDA_FM * loss_fm + total_loss

            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()

            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('fm_loss', loss_fm, on_epoch=True)
        else:
            # automatic_optimization is False, so a returned loss is discarded
            # by Lightning and nothing would train. Fail loudly instead.
            raise RuntimeError(
                "SynthesizerLightningModule requires a discriminator: with "
                "automatic_optimization=False a returned loss is never stepped."
            )

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return

        y_audio, y_hat_audio, mel_loss = self.shared_step(batch)
        batch_size = y_audio.size(0)

        # Every epoch, not every 5th, so val_emo_ccc can serve as a checkpoint
        # monitor: a monitor absent on most epochs makes the callback skip them.
        with torch.no_grad():
            self.y_hat_emo_list.append(self.ser(y_hat_audio).to(self.val_ccc.device))
            self.y_emo_list.append(self.ser(y_audio).to(self.val_ccc.device))

        if self.current_epoch % 5 == 0 and batch_idx == 0 and self._has_audio_logger():
            self.logger.experiment.add_audio(
                f"val_audio_{self.current_epoch}", y_hat_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )
            self.logger.experiment.add_audio(
                f"val_gt_audio_{self.current_epoch}", y_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )

        self.log('val_loss', mel_loss * LAMBDA_RECON, on_step=False, on_epoch=True,
                 batch_size=batch_size)
        return mel_loss

    def on_validation_epoch_end(self):
        if not self.y_hat_emo_list:
            return
        y_hat_emo_all = torch.cat(self.y_hat_emo_list, dim=0)
        y_emo_all = torch.cat(self.y_emo_list, dim=0)
        ccc_value = self.val_ccc(y_hat_emo_all, y_emo_all)
        ccc_loss = 1 - ccc_value

        self.log('val_ccc_arousal',   ccc_loss[0].item(), on_epoch=True)
        self.log('val_ccc_valence',   ccc_loss[1].item(), on_epoch=True)
        self.log('val_ccc_dominance', ccc_loss[2].item(), on_epoch=True)
        self.log('val_ccc_mean',      ccc_loss.mean().item(), on_epoch=True)

        # A torchmetrics Metric accumulates until reset; without this the logged
        # CCC silently pools every epoch seen so far.
        self.val_ccc.reset()
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
            # Extract style from a center 2.5s window, matching training distribution
            win_mel = self.segment_size * (16000 // 50) // 256  # ~156 frames
            center = mel_spec.shape[1] // 2
            half = win_mel // 2
            start = max(0, center - half)
            end = min(mel_spec.shape[1], start + win_mel)
            start = max(0, end - win_mel)
            mel_win = mel_spec[:, start:end, :]
            style = self.style_encoder(mel_win)
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
        base_lr = self.config['training']['learning_rate']
        groups = [
            {'params': self.decoder.parameters(), 'lr': base_lr},
            {'params': self.dict.parameters(),    'lr': base_lr},
        ]
        if not self.freeze_style_encoder:
            # Stage 2 only: joins at 0.1x the decoder LR.
            groups.insert(1, {'params': self.style_encoder.parameters(),
                              'lr': base_lr * 0.1})
        g_optimizer = torch.optim.AdamW(
            groups,
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
