import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path

class SpeechCommandsDataset(Dataset):
    """
    Handles 12-class logic (10 words + unknown + silence) and official Kaggle splitting.
    """
    
    TARGET_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    
    def __init__(self, root_dir = './/data//train', split='train', n_mels=64, apply_augment=False, time_mask=20, freq_mask=8):
        """
        Args:
            root_dir (str): Path to 'train' folder containing subfolders.
            split (str): 'train', 'val', or 'test'.
            n_mels (int): Mel-bands for spectrogram.
            apply_augment (bool): Only for train split.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.apply_augment = apply_augment
        self.sample_rate = 16000
        
        self.class_to_idx = {word: i for i, word in enumerate(self.TARGET_WORDS)}
        self.class_to_idx['unknown'] = 10
        self.class_to_idx['silence'] = 11

        with open(self.root_dir / 'validation_list.txt', 'r') as f:
            val_files = set(f.read().splitlines())
        with open(self.root_dir / 'testing_list.txt', 'r') as f:
            test_files = set(f.read().splitlines())
            
        self.file_paths = []
        self.labels = []
        self.bg_noise_files = []

        for label_dir in self.root_dir.iterdir():
            if not label_dir.is_dir(): continue
            
            label = label_dir.name
            
            if label == '_background_noise_':
                self.bg_noise_files = list(label_dir.glob('*.wav'))
                continue
            
            mapped_label = label if label in self.TARGET_WORDS else 'unknown'
            label_idx = self.class_to_idx[mapped_label]
            
            for wav_path in label_dir.glob('*.wav'):
                rel_path = f"{label}/{wav_path.name}"
                
                is_val = rel_path in val_files
                is_test = rel_path in test_files
                
                if split == 'val' and is_val:
                    self.file_paths.append(wav_path)
                    self.labels.append(label_idx)
                elif split == 'test' and is_test:
                    self.file_paths.append(wav_path)
                    self.labels.append(label_idx)
                elif split == 'train' and not is_val and not is_test:
                    self.file_paths.append(wav_path)
                    self.labels.append(label_idx)
        if split != 'test': 
            num_silence = 2300 if split == 'train' else 250 
            for _ in range(num_silence):
                self.file_paths.append("SILENCE_MARKER")
                self.labels.append(self.class_to_idx['silence'])

        self.mel_spec = T.MelSpectrogram(sample_rate=16000, n_mels=n_mels)
        self.amp_to_db = T.AmplitudeToDB()
        if self.apply_augment and split == 'train':
            self.spec_aug = torch.nn.Sequential(
            T.TimeMasking(time_mask_param=time_mask),
            T.FrequencyMasking(freq_mask_param=freq_mask)
        )

    def _get_random_silence(self):
        """Extracts a random 1s chunk from background noise files."""
        random_bg = random.choice(self.bg_noise_files)
        waveform, _ = torchaudio.load(random_bg)
        start = random.randint(0, waveform.shape[1] - 16001)
        return waveform[:, start:start+16000]

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        if path == "SILENCE_MARKER":
            waveform = self._get_random_silence()
        else:
            waveform, _ = torchaudio.load(path)
            # Pad/Truncate to 1s
            if waveform.shape[1] < 16000:
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
            else:
                waveform = waveform[:, :16000]
        
        # Transform
        spec = self.amp_to_db(self.mel_spec(waveform))
        
        # Augment
        if self.apply_augment and self.split == 'train':
            spec = self.spec_aug(spec)
            
        return spec, label

    def __len__(self):
        return len(self.file_paths)