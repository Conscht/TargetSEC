"""
HiFiGAN baseline synthesizer module.

Mirrors the TargetSEC synthesizer but replaces the style encoder with a
direct emotion embedding projection — matching the HiFiGAN [14] baseline:
  decoder input = HuBERT (128) + speaker (512→broadcast) + projected emotion (128)

No style encoder is used at any point (train or inference). This allows a
fair subjective comparison against TargetSEC in the human evaluation study.
"""
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

# Loss weights from Prabhu et al. (arXiv 2306.01916), eq. (6):
#   L_G = sum_j [ L_adv(D_j, G) + lambda_fm * L_fm(D_j, G) ] + lambda_r * L_recon + lambda_SER * L_SER
LAMBDA_FM = 2.0
LAMBDA_RECON = 45.0
LAMBDA_SER = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HiFiGANBaselineLightningModule(pl.LightningModule):
    """Direct emotion injection baseline (no style encoder, no LDM).

    At training and inference: emotion embedding (1024-dim) is projected to
    128-dim and used in place of the style vector.
    """

    def __init__(self, decoder, config, discriminator=None,
                 ser_loss_mode="differentiable"):
        """
        ser_loss_mode:
          "differentiable" -- L_SER backpropagates through the frozen SER into
              the generator. This is eq. (5) of the paper taken literally, but
              it creates a closed loop with the evaluation metric: the same
              audeering model scores the benchmark. Measured consequence --
              L_abs 0.052 against forced targets, versus 0.098 for the SER's
              own agreement with human labels on *real* speech. Beating the
              metric's noise floor is the signature of optimising the scorer,
              not the audio.
          "detached" -- computed and logged, no gradient. Conditioning still
              works: z_e is concatenated at every frame and the reconstruction
              loss teaches the decoder what the scalar means. Reproduces the
              regime the original paper's numbers are consistent with (their
              L_SER improved L_abs only 0.2642 -> 0.2442, i.e. 7.6%, which a
              live gradient through the scorer cannot produce).
        """
        super().__init__()
        assert ser_loss_mode in ("differentiable", "detached")
        self.decoder = decoder
        self.discriminator = discriminator
        self.config = config
        self.ser_loss_mode = ser_loss_mode
        self.automatic_optimization = False

        self.audio_time = 2.5
        self.segment_size = 125    # HuBERT frames for 2.5s @ 50 Hz
        self.dict = nn.Embedding(100, 128)
        # Scalar arousal [0,1] → 128-dim style slot: small MLP matching [7]'s "linear layers"
        self.emo_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        # Frozen SER for L_SER. Held in a plain list so nn.Module never registers
        # it: it must not enter the state_dict (it would add ~1.2 GB to every
        # checkpoint) and must not be flipped out of eval by .train().
        self._ser_holder = [DifferentiableSER()]

        self.test_outputs = []
        self.y_hat_emo_list = []
        self.y_emo_list = []
        # Validation-only metric. Kept separate from the training loss (which uses
        # the stateless concordance_cc) because a torchmetrics Metric accumulates
        # state across every call until reset -- sharing one instance between
        # train and val silently pools both into the logged value.
        self.val_ccc = ConcordanceCorrCoef(num_outputs=1)

    @property
    def ser(self):
        """Frozen SER, moved to the module's device on first use."""
        m = self._ser_holder[0]
        if next(m.parameters()).device != self.device:
            m.to(self.device)
        return m

    def _has_audio_logger(self):
        """True when a TensorBoard-style logger with add_audio is attached.

        Guards against logger=False (fast_dev_run, smoke tests) and against
        loggers such as CSVLogger that have no add_audio.
        """
        logger = self.logger
        return (logger is not None
                and getattr(logger, "experiment", None) is not None
                and hasattr(logger.experiment, "add_audio"))

    def slice_audio(self, raw_audio, start_ids):
        """Cut the 2.5 s waveform window matching the sampled HuBERT frames.

        The mel target is no longer sliced from the cache -- it is recomputed
        from this same window in shared_step, which is what removes the
        320-vs-256 stride misalignment the cached path had.
        """
        sr = 16000
        hubert_rate = 50.0
        win_audio = int(self.audio_time * sr)

        B = raw_audio.size(0)
        T_audio = raw_audio.size(1)

        audio_segments = []
        for i in range(B):
            s_h = start_ids[i].item()
            start_sample = int(s_h / hubert_rate * sr)
            end_sample = start_sample + win_audio

            if end_sample > T_audio:
                end_sample = T_audio
                start_sample = max(0, end_sample - win_audio)

            audio_seg = raw_audio[i, start_sample:end_sample]

            if audio_seg.size(0) < win_audio:
                pad_len = win_audio - audio_seg.size(0)
                audio_seg = F.pad(
                    audio_seg.unsqueeze(0).unsqueeze(0), (0, pad_len), mode='reflect'
                ).squeeze(0).squeeze(0)

            audio_segments.append(audio_seg.unsqueeze(0))

        return torch.cat(audio_segments, dim=0)

    def shared_step(self, batch):
        raw_audio = batch['audio']
        linguistic_emb, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            batch['hubert'], batch_first=True
        )
        speaker_emb = batch['speaker_emb']

        x_ids, start_id = rand_slice_segments(
            linguistic_emb, x_lengths=lengths, segment_size=self.segment_size
        )

        x = self.dict(x_ids).transpose(1, 2)

        y_audio = self.slice_audio(raw_audio, start_id)

        # Annotated arousal label (EmoAct-1)/6 ∈ [0,1] from labels_consensus.csv.
        # Same scale used at inference: (c-1)/6 for c ∈ {1,...,7} → never OOD.
        arousal_scalar = batch['arousal_scalar'].unsqueeze(1)  # (B,) → (B, 1)
        style_emb = self.emo_proj(arousal_scalar)   # (B, 128)

        x = broadcast_embeddings(x, speaker_emb, style_emb)
        y_hat_audio = self.decoder(x).squeeze(1)

        # Both mels come from the waveforms via the differentiable front-end.
        # Two reasons this replaces the cached-mel path:
        #   1. get_mel_from_wav returns a NumPy array, so the old mel_loss had
        #      requires_grad=False -- 45 * L_recon trained nothing.
        #   2. the cached mel was sliced at start_sample // 256 while the audio
        #      was sliced at s_h * 320; 320 is not a multiple of 256, so target
        #      and prediction were misaligned by up to one frame.
        y_hat_mel = mel_spectrogram_torch(y_hat_audio)
        y_mel = mel_spectrogram_torch(y_audio)
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

        # L_SER = 1 - CCC(e, E_SER(y_hat))  -- eq. (5) of the paper.
        #
        # Two things this fixes versus the previous implementation:
        #   1. It is differentiable. The old code called process_func on
        #      y_hat_audio.detach().cpu().numpy() under torch.no_grad(), so
        #      emo_loss was a constant and contributed no gradient at all.
        #   2. The target is `e`, the annotated arousal label, not
        #      E_SER(y_real). Comparing SER(y_hat) to SER(y_real) is a
        #      resynthesis-consistency term: it never asks the generator to
        #      honour the conditioning value, so z_e could be ignored entirely.
        #
        # CCC is a batch statistic, so this term is meaningless at batch size 1
        # and noisy below ~16.
        if self.ser_loss_mode == "detached":
            # Logged as a diagnostic only. A term with no gradient path leaves
            # every weight update unchanged, so this is mathematically the same
            # as omitting it -- the conditioning comes from z_e + L_recon.
            with torch.no_grad():
                pred_arousal = self.ser(y_hat_audio)[:, 0]
            target_arousal = batch['arousal_scalar'].to(pred_arousal.dtype)
            emo_loss = 1.0 - concordance_cc(target_arousal, pred_arousal)
            total_loss = LAMBDA_RECON * mel_loss
        else:
            pred_arousal = self.ser(y_hat_audio)[:, 0]      # arousal is column 0
            target_arousal = batch['arousal_scalar'].to(pred_arousal.dtype)
            emo_loss = 1.0 - concordance_cc(target_arousal, pred_arousal)
            total_loss = LAMBDA_RECON * mel_loss + LAMBDA_SER * emo_loss
        self.log('train_mel_loss', mel_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_ccc', emo_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_arousal_mae', (pred_arousal - target_arousal).abs().mean(),
                 on_epoch=True, batch_size=batch_size)

        if self.current_epoch % 5 == 0 and batch_idx == 0 and self._has_audio_logger():
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

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y_audio_d, y_hat_audio_d.detach())
            d_loss, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y_audio_d, y_hat_audio_d)
            # decoder_modules.feature_loss already multiplies by 2, so the old
            # `2 * feature_loss(...)` applied lambda_fm = 4, not the paper's 2.
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
                "HiFiGANBaselineLightningModule requires a discriminator: with "
                "automatic_optimization=False a returned loss is never stepped."
            )

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return

        y_audio, y_hat_audio, mel_loss = self.shared_step(batch)
        batch_size = y_audio.size(0)

        # Every epoch, not every 5th: val_arousal_mae is a ModelCheckpoint
        # monitor, and a monitor that is missing on most epochs makes the
        # callback skip them.
        with torch.no_grad():
            pred_arousal = self.ser(y_hat_audio)[:, 0]
        # Score against the annotated label, matching both the training loss
        # and the benchmark's L_abs/L_mse -- not against SER(y_real).
        self.y_hat_emo_list.append(pred_arousal.detach().to(self.val_ccc.device))
        self.y_emo_list.append(
            batch['arousal_scalar'].detach().to(self.val_ccc.device))

        if self.current_epoch % 5 == 0 and batch_idx == 0 and self._has_audio_logger():
            self.logger.experiment.add_audio(
                f"val_audio_{self.current_epoch}", y_hat_audio[0], self.current_epoch,
                sample_rate=self.config['data']['sampling_rate']
            )

        self.log('val_loss', mel_loss * LAMBDA_RECON, on_step=False, on_epoch=True,
                 batch_size=batch_size)
        return mel_loss

    def on_validation_epoch_end(self):
        if not self.y_hat_emo_list:
            return
        pred_all = torch.cat(self.y_hat_emo_list, dim=0).reshape(-1, 1)
        target_all = torch.cat(self.y_emo_list, dim=0).reshape(-1, 1)

        ccc_loss = 1 - self.val_ccc(pred_all, target_all)
        self.log('val_ccc_arousal', ccc_loss.item(), on_epoch=True)

        # These two are the paper's L_abs / L_mse, computed here on the
        # *source* arousal rather than swept over all 7 targets, so they track
        # the benchmark without being directly comparable to it.
        err = pred_all - target_all
        self.log('val_arousal_mae', err.abs().mean().item(), on_epoch=True)
        self.log('val_arousal_mse', err.pow(2).mean().item(), on_epoch=True)

        # A torchmetrics Metric keeps accumulating until reset; without this the
        # logged CCC silently pools every epoch seen so far.
        self.val_ccc.reset()
        self.y_hat_emo_list.clear()
        self.y_emo_list.clear()

    def test_step(self, batch, batch_idx):
        linguistic_emb, _ = torch.nn.utils.rnn.pad_packed_sequence(
            batch['hubert'], batch_first=True
        )
        speaker_emb = batch['speaker_emb']
        arousal_scalar = batch['arousal_scalar'].unsqueeze(1)  # (B, 1)
        style_emb = self.emo_proj(arousal_scalar)
        x = self.dict(linguistic_emb).transpose(1, 2)
        x = broadcast_embeddings(x, speaker_emb, style_emb)
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
        g_optimizer = torch.optim.AdamW(
            list(self.decoder.parameters()) + list(self.emo_proj.parameters()) + list(self.dict.parameters()),
            lr=base_lr,
            betas=(0.8, 0.99),
            weight_decay=0.01
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            g_optimizer, gamma=0.999**(1/8)
        )

        if self.discriminator:
            d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=base_lr,
                betas=(0.8, 0.99),
                weight_decay=0.01
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                d_optimizer, gamma=0.999**(1/8)
            )
            return [g_optimizer, d_optimizer], [scheduler_g, scheduler_d]
        else:
            return [g_optimizer], [scheduler_g]
