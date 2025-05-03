import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from torch.utils.data import DataLoader, random_split
import ast
from torch.nn.utils.rnn import pack_sequence


class MelSpectrogramDataset(Dataset):
    def __init__(self, tensor_directory, embedding_file="/Users/Conscht/Documents/New folder/Audio/MSP-Podcast-1.10/hubert-km100/parsed_with_spkrEmbeds/train.txt", transform=None):
        self.audio_directory =  '/Users/Conscht/Documents/New folder/Audio/Audio'
        self.arousal_data = 'empty'
        self.tensor_directory = tensor_directory
        self.transform = transform
        self.file_names = [f for f in os.listdir(tensor_directory) if f.endswith('_mel.pt')]
        self.embeddings = self.load_embeddings(embedding_file)
        self.expected_hubert_length = 75
        self.expected_mel_length = 32
        self.expected_audio_length = 32 * 256
        self.counter = 0
        self.skipped_samples = 0


        
    def load_embeddings(self, embedding_file):
        embeddings = {}
        with open(embedding_file, 'r') as f:
            for line in f:
                data = ast.literal_eval(line.strip())  # Convert string to dictionary
                audio_key = os.path.basename(data['audio'])  # Extract just the file nametmux
                embeddings[audio_key] = data
        return embeddings    

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        mel_file = os.path.join(self.tensor_directory, self.file_names[index])
        mel_spectrogram = torch.load(mel_file)



        if len(mel_spectrogram.shape) == 3 and mel_spectrogram.shape[0] == 1:
            mel_spectrogram = mel_spectrogram.squeeze(0)

            

        audio_file = os.path.join(self.audio_directory, self.file_names[index].replace('_mel.pt', '.wav'))
        audio = self.load_audio(audio_file)

        # Find the matching embeddings based on the audio file name
        audio_key = os.path.basename(audio_file)
        if audio_key not in self.embeddings:
            raise KeyError(f"Embedding for {audio_key} not found.")

        hubert_embedding = torch.tensor([int(x) for x in self.embeddings[audio_key]['hubert'].split()])
        if hubert_embedding.size(0) < self.expected_hubert_length:
            self.counter = self.counter+1
            print(self.counter, "many elements have been removes")
            return None  # Skip this entry by returning None
        
        if mel_spectrogram.shape[1] < self.expected_mel_length:
            self.skipped_samples += 1
            return None
        if audio.size(0) < self.expected_audio_length:
            self.skipped_samples += 1
            return None

        speaker_embedding = torch.tensor(self.embeddings[audio_key]['spkr_embeds'])

        #  (mel_channels, time_steps) => Exptected of Style Encoder (time_steps, mel_channels) 
        mel_spectrogram = mel_spectrogram.transpose(1, 0)  # (80, 128) =>(128, 80)
        sample = {
            'mel_spectrogram': mel_spectrogram,
            'audio': audio,
            'hubert': hubert_embedding,
            'speaker_emb': speaker_embedding,
            'arousal': None
        }

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
    

    def load_audio(self, file_path):
        try:
            audio, _ = librosa.load(file_path, sr=None)  
            audio_tensor = torch.tensor(audio, dtype=torch.float32) 
            return audio_tensor
        except Exception as e:
            print(f"Failed to load file {file_path}: {str(e)}")
            raise


def collate_fn(batch):
    """Simple collate function to build the batches."""
    batch = [item for item in batch if item is not None]

    # Unpack the batch data
    mel_spectrograms = [item['mel_spectrogram'] for item in batch]
    audios = [item['audio'] for item in batch]
    huberts = [item['hubert'] for item in batch]
    speaker_embs = [item['speaker_emb'] for item in batch]

    # Calculate max length of audio samples
    max_length = max(audio.shape[0] for audio in audios)
    max_length_mel = max(mel_spec.shape[0] for mel_spec in mel_spectrograms)

    # Safe length for slice method
    mel_original_lengths = [mel_spec.shape[0] for mel_spec in mel_spectrograms]
    audio_original_lengths = [audio.shape[0] for audio in audios]


    

    # Pad audio samples to max_length
    padded_audios = [torch.cat((audio, torch.zeros(max_length - audio.shape[0]))) if audio.shape[0] < max_length else audio for audio in audios]
    # Huberts can be variable length, since we slice it appropiatly inside the training

    mel_spectrograms = [torch.from_numpy(mel) if isinstance(mel, np.ndarray) else mel for mel in mel_spectrograms]
    expected_hubert_length = 75

    packed_huberts = pack_sequence(huberts, enforce_sorted=False)

    padded_mels = [torch.cat((mel_spec, torch.zeros((max_length_mel - mel_spec.shape[0], mel_spec.shape[1])))) if mel_spec.shape[0] < max_length_mel else mel_spec for mel_spec in mel_spectrograms]

    audio_attention_mask = [torch.cat((torch.ones(length), torch.zeros(max_length - length))) for length in audio_original_lengths]
    audio_attention_mask = torch.stack(audio_attention_mask)

    return {
        'mel_spectrogram': torch.stack(padded_mels),
        'audio': torch.stack(padded_audios),
        'hubert': packed_huberts,
        'speaker_emb': torch.stack(speaker_embs),
        'arousal' : NotImplemented,
        'mel_original_lengths': mel_original_lengths,
        'audio_attention_mask': audio_attention_mask

        
    }


def create_dataloaders(batch_size, val_split=0.2):
    tensor_directory = '/Users/Conscht/Documents/New folder/mel_spectograms/Train'

    full_dataset = MelSpectrogramDataset(tensor_directory=tensor_directory)

    print(f"[INFO] Skipped samples during dataset construction: {full_dataset.skipped_samples}")

    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        persistent_workers=True,
        collate_fn=collate_fn 
    )
    
    return train_loader, val_loader

def test_create_data_loader(batch_size, val_split=0.2):
    tensor_directory = '/Users/Conscht/Documents/New folder/mel_spectograms/Test1'


    full_dataset = MelSpectrogramDataset(tensor_directory=tensor_directory)
    torch.manual_seed(42)
    test_size = int(len(full_dataset))
    _ = 0
    test_dataset, _ = random_split(full_dataset, [test_size, _])


    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn  
    )

    
    return test_loader



import os
import random
import shutil

def split_dataset(input_directory, output_directory, train_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
        input_directory (str): Path to the directory containing the dataset.
        output_directory (str): Path to the directory where the split datasets will be stored.
        train_ratio (float): Ratio of the dataset to be used for training. The rest will be used for validation.
    """
    # Get all files in the input directory
    all_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    random.shuffle(all_files)
    
    # Calculate the number of training samples
    num_train = int(len(all_files) * train_ratio)
    
    # Split the files into training and validation sets
    train_files = all_files[:num_train]
    val_files = all_files[num_train:]
    
    # Create output directories for train and val sets
    train_directory = os.path.join(output_directory, 'train')
    val_directory = os.path.join(output_directory, 'val')
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)
    
    # Move the files to the respective directories
    for f in train_files:
        shutil.move(os.path.join(input_directory, f), os.path.join(train_directory, f))
    for f in val_files:
        shutil.move(os.path.join(input_directory, f), os.path.join(val_directory, f))

    print(f'Training files: {len(train_files)}')
    print(f'Validation files: {len(val_files)}')


