import torch
import shutil
import os
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from processing.stft import TacotronSTFT
from config.stylespeech_model_config import style_config as config


# _stft used as inStyle Speech
_stft = TacotronSTFT(
            config.filter_length,
            config.hop_length,
            config.win_length,
            config.n_mel_channels,
            config.sampling_rate,
            config.mel_fmin,
            config.mel_fmax)

device = 'cuda'



# Schritt 3: Funktionen zur Umwandlung in Mel-Spektrogramme => convert file into may length
def truncate_or_pad(tensor, target_length):
    current_length = tensor.size(-1)
    if current_length > target_length:
        tensor = tensor[:, :, :target_length]
    elif current_length < target_length:
        pad_size = target_length - current_length
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))
    return tensor


def get_mel_from_wav(audio, _stft):
    # print(f"y_hat_audio device: {audio.device}, audio device: {audio.device}")
    if isinstance(audio, torch.Tensor):
        audio = torch.clip(audio.float().to(device), -1, 1)
    else:
        audio = torch.clip(torch.tensor(audio, dtype=torch.float32, device=device), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)


    return melspec

def audio_to_mel(path_to_audio, save_directory, sr=16000, max_seq_len=128):
    waveform, sample_rate = torchaudio.load(path_to_audio)
    
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)


    mel_spectrogram = get_mel_from_wav(waveform, _stft=_stft)

    

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    file_name = os.path.splitext(os.path.basename(path_to_audio))[0] + '_mel.pt'
    torch.save(mel_spectrogram, os.path.join(save_directory, file_name))

def process_directory(audio_directory, save_directory, sr=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmax=8000, max_seq_len=128):
    for root, _, files in os.walk(audio_directory):
        for file_name in files:
            if file_name.lower().endswith('.wav'):
                path_to_audio = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, audio_directory)
                save_dir = os.path.join(save_directory, relative_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                audio_to_mel(path_to_audio, save_dir, sr, n_fft, hop_length, win_length, n_mels, fmax, max_seq_len)
                print(f'Processed {file_name}')

# audio_directory = '/export/home/1auga/infhome/Documents/ba-constantin-ldm/audio'
# save_directory = '/export/home/1auga/infhome/Documents/ba-constantin-ldm/mel_spectrograms'
# process_directory(audio_directory, save_directory)



