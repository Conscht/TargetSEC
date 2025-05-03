from src.decoder.decoder_modules import rand_slice_segments, broadcast_single_embedding, broadcast_embeddings, feature_loss, discriminator_loss, generator_loss
import torch
from torch import nn
from processing.preprocessor import get_mel_from_wav, _stft
#! Version is still recent
import io

import torch.distributed as dist

import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from test_audio import calculate_vmos
from src.emotion.emotion_encoder import process_func
from torchmetrics.regression  import ConcordanceCorrCoef 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'

class SynthesizerLightningModule(pl.LightningModule):
    def __init__(self, style_encoder, decoder, config, discriminator=None):
        super(SynthesizerLightningModule, self).__init__()
        self.style_encoder = style_encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.config = config
        self.automatic_optimization = False # Needed because of Discrimminator usage
        self.audio_time = 1.5
        self.segment_size = 75
        self.dict = nn.Embedding(100, 128)
        self.stft = _stft.to(device=device)
        self.val_generated_audios = []
        self.test_outputs = []
        self.y_hat_emo_list = []
        self.y_emo_list = []
        # self.validation_step_outputs = []

        self.ccc = ConcordanceCorrCoef(num_outputs=3)

        


    def forward(self, hubert, speaker, audio, emo):
        style_emb = diffusion(emo)
        x = self.dict(hubert).transpose(1, 2)
        x = broadcast_embeddings(x, speaker, style_emb)

        audio_output = self.decoder(style_emb)
        return audio_output
        

    def training_step(self, batch, batch_idx):
        raw_audio = batch['audio']
        mel_spec = batch['mel_spectrogram']
        linguistic_emb = batch['hubert']
        speaker_emb = batch['speaker_emb']
        emo_loss = True
        batch_size = raw_audio.size(0)

        linguistic_emb, lengths = torch.nn.utils.rnn.pad_packed_sequence(linguistic_emb, batch_first=True)



    

        audio_start_ids = ((start_id.long() / 50) * 16000).long()  
        audio_end_ids = (audio_start_ids + self.audio_time * 16000).long() 

        y_audio = []
        for i in range(raw_audio.size(0)): 
            start_id = audio_start_ids[i].item() 
            end_id = audio_end_ids[i].item() 
            y_audio.append(raw_audio[i, start_id:end_id])  
        y_audio = torch.stack(y_audio)


        # Calculate the corresponding mel spectrogram slice
        mel_start_id = (audio_start_ids // 256).long()
        mel_end_id = (audio_end_ids // 256).long()

        # Slice the corresponding mel spectrogram segment
        mel_spec_segment = []
        for i in range(mel_spec.size(0)):
            mel_spec_segment.append(mel_spec[i, mel_start_id[i]:mel_end_id[i], :])

        target_length = min(segment.size(0) for segment in mel_spec_segment)   
        # Truncate all segments to the target length
        mel_spec_segment = [segment[:target_length, :] for segment in mel_spec_segment]
     
        mel_spec_segment = torch.stack(mel_spec_segment)
        # print(mel_spec_segment.shape) # torch.Size([16, 31, 80])

        with torch.no_grad():
            style_emb = self.style_encoder(mel_spec_segment)

        y_mel = mel_spec_segment #[16, 31, 80])


        x = broadcast_embeddings(x, speaker_emb, style_emb)
        # x = broadcast_single_embedding(x, style_emb=speaker_emb)

        y_hat_audio = self.decoder(x).to(device) # batch, 1, timesteps

        y_hat_audio = y_hat_audio.squeeze(1)
        y_hat_mel = get_mel_from_wav(y_hat_audio, self.stft)

        y_hat_mel = torch.from_numpy(y_hat_mel).to(y_mel.device) # [16, 80, 32]

        # Adjust y_hat_audio shape to match y_audio
        # remove extra dimension

        y_hat_audio = y_hat_audio.to(y_audio.device)
 
        # Ensure y_audio and y_hat_audio have the same size, this is for the batch size 1 case
        if y_audio.shape[-1] != y_hat_audio.shape[-1]:
            min_length = min(y_audio.shape[-1], y_hat_audio.shape[-1])
            y_audio = y_audio[..., :min_length]
            y_hat_audio = y_hat_audio[..., :min_length]

      # Adjust y_hat_audio shape to match y_audio
        y_mel = y_mel.squeeze(0).transpose(1, 2)
        # print("y_mel", y_mel.shape)
        # Ensure y_mel and y_hat_mel have the same size, this is for the batch size 1 case
        if y_mel.shape[-1] != y_hat_mel.shape[-1]:
            min_length = min(y_mel.shape[-1], y_hat_mel.shape[-1])
            y_mel = y_mel[..., :min_length]
            y_hat_mel = y_hat_mel[..., :min_length]

     

        mel_loss = F.l1_loss(y_hat_mel.transpose(1, 2), y_mel.transpose(1, 2)) # (batch, timesteps , bins) => (batch, bins , timesteps)
        # print("in training, y audio", y_audio.shape, y_hat_audio.shape)
        if self.current_epoch % 5 == 0 and batch_idx == 0:
            # Plot the Audio
            self.logger.experiment.add_audio(f'train_gen_epoch{self.current_epoch}', y_hat_audio[0], self.current_epoch, self.config['data']['sampling_rate'])
            self.logger.experiment.add_audio(f'train_truth_epoch{self.current_epoch}', y_audio[0], self.current_epoch, self.config['data']['sampling_rate'])

            # Plot the Mel Spec
            # gen_mel_img = self.plot_mel_spectrogram(y_hat_mel[0].detach().cpu())
            # truth_mel_img = self.plot_mel_spectrogram(y_mel[0].detach().cpu())
            # self.logger.experiment.add_image(f'train_gen_mel_epoch{self.current_epoch}', gen_mel_img, self.current_epoch)
            # self.logger.experiment.add_image(f'train_truth_mel_epoch{self.current_epoch}', truth_mel_img, self.current_epoch)

        y_audio = y_audio.unsqueeze(1) # from (batch, timesteps) =>  (batch, 1, timesteps) for the Discrimminator Conv1D 
        y_hat_audio = y_hat_audio.unsqueeze(1) # from (batch, timesteps) =>  (batch, 1, timesteps) for the Discrimminator Conv1D 
        self.log('train_mel_loss', mel_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        if self.discriminator is not None:
            #Setup Generator and Discrimminator optimizers
            optimizer_g, optimizer_d = self.optimizers()

            # Discriminator step
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y_audio, y_hat_audio.detach())

            # Discrimminator Loss LSGAN
            d_loss, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            
            # Optimize Discrimminator
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)

            optimizer_d.step()

            # Generator step

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y_audio, y_hat_audio)

            # Generator loss LSGAN (Adversial + FM + Mel-Spec)
            loss_fm = feature_loss(fmap_r, fmap_g)
            g_adv_loss, _ = generator_loss(y_d_hat_g)
            g_loss =  g_adv_loss + loss_fm + mel_loss 
            
            if emo_loss:
                y_hat_audio = y_hat_audio.squeeze(1)
                y_audio = y_audio.squeeze(1)

                y_hat_emo = process_func(y_hat_audio.detach().cpu().numpy(), device=linguistic_emb.device, embeddings=False)
                y_emo = process_func(y_audio.detach().cpu().numpy(), device=linguistic_emb.device, embeddings=False)
                # Compute CCC separately for Arousal, Dominance, and Valence
                # y_hat_emo = torch.from_numpy(y_hat_emo).clone().detach().to(self.ccc.device)
                # y_emo = torch.from_numpy(y_emo).clone().detach().to(self.ccc.device)
                ccc_value = self.ccc(y_hat_emo.to(self.ccc.device), y_emo.to(self.ccc.device))

                # CCC loss: 1 - CCC for each emotion dimension
                emo_loss = 1 - ccc_value

                g_loss = g_adv_loss + 2 * loss_fm + 45 * mel_loss + emo_loss.mean()
                # Log CCC separately for each emotion dimension (Arousal, Valence, Dominance)
                self.log('val_ccc_arousal', emo_loss[0].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
                self.log('val_ccc_valence', emo_loss[1].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
                self.log('val_ccc_dominance', emo_loss[2].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
                self.log('train_ccc', emo_loss.mean(), on_epoch=True, logger=True, batch_size=batch_size, sync_dist=True)



            else:
                g_loss = g_adv_loss + 2 * loss_fm + 45 * mel_loss

            # Optimize Generator
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)

            optimizer_g.step()


            # Logging losses
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            # without Discrimminator we have our casual optimizer
            total_loss = mel_loss
            return total_loss
  


    def validation_step(self, batch, batch_idx):
        raw_audio = batch['audio']
        mel_spec = batch['mel_spectrogram']
        linguistic_emb = batch['hubert']
        speaker_emb = batch['speaker_emb']
        emo_loss = True

        batch_size = next(iter(batch.values())).size(0)
#
  
        linguistic_emb, lengths = torch.nn.utils.rnn.pad_packed_sequence(linguistic_emb, batch_first=True)

        lengths = lengths.to(device=linguistic_emb.device)  # Ensure lengths is on the same device


        if batch_size > 1:
            x, start_id = rand_slice_segments(linguistic_emb, x_lengths=lengths, segment_size=self.segment_size)
            audio_start_ids = ((start_id.long() / 50) * 16000).long()
            audio_end_ids = (audio_start_ids + self.audio_time * 16000).long()

            y_audio = []
            for i in range(raw_audio.size(0)):
                start_id = audio_start_ids[i].item()
                end_id = audio_end_ids[i].item()
                y_audio.append(raw_audio[i, start_id:end_id])
            y_audio = torch.stack(y_audio)
        else:
            x, start_id = rand_slice_segments(linguistic_emb, x_lengths=lengths, segment_size=lengths)
            audio_start_ids = ((start_id.long() / 50) * 16000).long()
            audio_end_ids = (audio_start_ids + (lengths / 50) * 16000).long()

            y_audio = []
            for i in range(raw_audio.size(0)):
                start_id = audio_start_ids[i].item()
                end_id = audio_end_ids[i].item()
                y_audio.append(raw_audio[i, start_id:end_id])
            y_audio = torch.stack(y_audio)

        x = self.dict(x).transpose(1, 2)

        mel_start_id = (audio_start_ids // 256).long()
        mel_end_id = (audio_end_ids // 256).long()

        # Slice the corresponding mel spectrogram segment
        mel_spec_segment = []
        for i in range(mel_spec.size(0)):
            mel_spec_segment.append(mel_spec[i, mel_start_id[i]:mel_end_id[i], :])

        target_length = min(segment.size(0) for segment in mel_spec_segment)   
        # Truncate all segments to the target length
        mel_spec_segment = [segment[:target_length, :] for segment in mel_spec_segment]
     
        mel_spec_segment = torch.stack(mel_spec_segment)
        with torch.no_grad():
            style_emb = self.style_encoder(mel_spec_segment)

        y_mel = mel_spec_segment

        x = broadcast_embeddings(x, speaker_emb, style_emb)
        # x = broadcast_single_embedding(x, style_emb=emb)

        y_hat_audio = self.decoder(x).to(device)
 
         # Ensure y_audio and y_hat_audio have the same size, this is for the batch size 1 case
        if y_audio.shape[-1] != y_hat_audio.shape[-1]:
            min_length = min(y_audio.shape[-1], y_hat_audio.shape[-1])
            y_audio = y_audio[..., :min_length]
            y_hat_audio = y_hat_audio[..., :min_length]

        y_hat_audio = y_hat_audio.squeeze(1)  # batch, 1, timesteps => batch, timesteps
        y_hat_mel = get_mel_from_wav(y_hat_audio, self.stft)
        y_hat_mel = torch.from_numpy(y_hat_mel).to(y_mel.device)


        # Adjust y_hat_audio shape to match y_audio
        y_mel = y_mel.squeeze(0).transpose(0, 1) #  80, timesteps

        if y_mel.shape[-1] != y_hat_mel.shape[-1]:
            min_length = min(y_mel.shape[-1], y_hat_mel.shape[-1])
            y_mel = y_mel[..., :min_length]
            y_hat_mel = y_hat_mel[..., :min_length]


        #print("In val audio", y_hat_audio.shape, y_audio.shape) # ([1, 171840])
        mel_loss = F.l1_loss(y_hat_mel, y_mel) # (Bins, Timesteps)

        if emo_loss:
            y_hat_emo = process_func(y_hat_audio.cpu().numpy(), device=linguistic_emb.device, embeddings=False)
            y_emo = process_func(y_audio.cpu().numpy(), device=linguistic_emb.device, embeddings=False)
            # Convert NumPy arrays to tensors
            # y_hat_emo = torch.from_numpy(y_hat_emo).clone().detach().to(self.ccc.device)
            # y_emo = torch.from_numpy(y_emo).clone().detach().to(self.ccc.device)
            # Collect the predictions and targets for CCC computation
            self.y_hat_emo_list.append(y_hat_emo.to(self.ccc.device))
            self.y_emo_list.append(y_emo.to(self.ccc.device))
            
            ccc_value = self.ccc(y_hat_emo.to(self.ccc.device), y_emo.to(self.ccc.device))

            emo_loss = 1 - ccc_value


            total_loss = mel_loss * 45 + emo_loss.mean()
        else: 
            total_loss = mel_loss * 45 



       # y_hat_audio[0] = y_hat_audio[0].to(self.device)
        if self.current_epoch % 5 == 0 and batch_idx == 0:
            # Plot the Mel Spec
            self.logger.experiment.add_audio(f'val_gen_epoch{self.current_epoch}', y_hat_audio[0], self.current_epoch, self.config['data']['sampling_rate'])
            self.logger.experiment.add_audio(f'val_truth_sample_epoch{self.current_epoch}', y_audio[0], self.current_epoch, self.config['data']['sampling_rate'])
        # self.validation_step_outputs.append(total_loss)


        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)    
        return total_loss

    # def on_validation_epoch_end(self):
    #     # After all validation steps, accumulate the predictions and targets
    #     y_hat_emo_all = torch.cat(self.y_hat_emo_list, dim=0)
    #     y_emo_all = torch.cat(self.y_emo_list, dim=0)

    #     # Compute CCC for all accumulated predictions and targets
    #     ccc_value = self.ccc(y_hat_emo_all, y_emo_all)
    #     ccc_loss = 1 - ccc_value  # CCC loss for the entire epoch

    #     # Log CCC separately for each emotion dimension (Arousal, Valence, Dominance)
    #     self.log('val_ccc_arousal', ccc_loss[0].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #     self.log('val_ccc_valence', ccc_loss[1].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    #     self.log('val_ccc_dominance', ccc_loss[2].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #     # Optionally, you can also log the mean CCC value
    #     total_ccc_loss = ccc_loss.mean()  # Better clarity and correctness

    #     self.log('val_ccc_epoch', total_ccc_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
    #     # Log the average CCC for the entire validation epoch

    #     loss = torch.stack(self.validation_step_outputs).mean()
    #     total_loss = loss + total_ccc_loss  # You can add CCC loss to the overall validation loss

    #     # Log the total validation loss
    #     self.log('val_epoch_loss', total_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    #     # Clear the lists for the next epoch
    #     self.validation_step_outputs.clear() 

    #     self.y_hat_emo_list.clear()
    #     self.y_emo_list.clear()

    def test_step(self, batch, batch_idx):
        mel_spec = batch['mel_spectrogram']
        linguistic_emb = batch['hubert']
        speaker_emb = batch['speaker_emb']

        linguistic_emb, _ = torch.nn.utils.rnn.pad_packed_sequence(linguistic_emb, batch_first=True)
        linguistic_emb = self.dict(linguistic_emb).transpose(1, 2)

        with torch.no_grad():
            style = self.style_encoder(mel_spec)

        embedding = broadcast_embeddings(linguistic_emb, speaker_emb, style)

        output = self.decoder(embedding)

        vmos = calculate_vmos(output.cpu(), batch_idx)

        
        self.test_outputs.append({'loss': vmos})

        return {'loss': vmos}
    
    def on_test_epoch_end(self):
        """Function to aggregate and log results after the test epoch."""
        # You can compute the average VMOS here or other metrics
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        # Compute any additional metrics from the stored test outputs
        # e.g., calculate average VMOS or accuracy
        print(f"Test Completed - Average Loss: {avg_loss.item()}")

        # Clear the stored outputs for the next run
        self.test_outputs.clear()


    #! lets try VITS optimizerS
    def configure_optimizers(self):
        """Configure the Optimizer and Scheduler of Generatore and Discrimminator if existing.
        
        Adjust  learning rate of Generator / Discrimminator if needed.

        Returns:
        (tuple) optimizers: the Optimizer and Scheduler of Generator and or Discrimminator
        """
        g_optimizer = torch.optim.AdamW(
            self.decoder.parameters(), 
            lr=self.config['training']['learning_rate'],  #*0.75
            betas=(0.8, 0.99), 
            weight_decay=0.01
        )
        
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            g_optimizer, 
            gamma=0.999**(1/8)
        )
        
        if self.discriminator:
            d_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(), 
                lr=self.config['training']['learning_rate'], #*0.5*0.5
                betas=(0.8, 0.99), 
                weight_decay=0.01
            )
            
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                d_optimizer, 
                gamma=0.999**(1/8)
            )
            
            return [g_optimizer, d_optimizer], [scheduler_g, scheduler_d]
        else:
            return [g_optimizer], [scheduler_g]

