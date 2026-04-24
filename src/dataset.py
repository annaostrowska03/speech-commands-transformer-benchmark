import random
import warnings
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from torch.utils.data import Dataset

class SpeechCommandsDataset(Dataset):
    """Speech Commands v1 dataset with 12-class mapping and official val/test split files."""

    TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    SILENCE_MARKER = "__silence__"

    def __init__(
        self,
        root_dir="./data/train",
        split="train",
        n_mels=64,
        apply_augment=False,
        time_mask=20,
        freq_mask=8,
        silence_train_samples=2300,
        silence_eval_samples=250,
        include_silence_in_test=False,
        silence_test_samples=None,
        unknown_keep_prob=1.0,
    ):
        """
        Args:
            root_dir (str): Path to train folder containing class subfolders.
            split (str): One of: train, val, test.
            n_mels (int): Number of mel bands.
            apply_augment (bool): Enable SpecAugment only on train split.
            time_mask (int): Max width for time masking.
            freq_mask (int): Max width for frequency masking.
            silence_train_samples (int): Number of synthetic silence samples for train split.
            silence_eval_samples (int): Number of synthetic silence samples for val split.
            include_silence_in_test (bool): If True, include synthetic silence also in test split.
            silence_test_samples (int | None): Number of synthetic silence samples for test split when enabled.
            unknown_keep_prob (float): Probability of keeping an "unknown" sample in train split.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not (0.0 < unknown_keep_prob <= 1.0):
            raise ValueError("unknown_keep_prob must be in (0, 1]")
        if silence_train_samples < 0 or silence_eval_samples < 0:
            raise ValueError("silence sample counts must be non-negative")
        if silence_test_samples is not None and silence_test_samples < 0:
            raise ValueError("silence_test_samples must be non-negative when provided")

        self.root_dir = Path(root_dir)
        self.audio_root = self.root_dir / "audio" if (self.root_dir / "audio").exists() else self.root_dir
        self.split = split
        self.sample_rate = 16000
        self.apply_augment = apply_augment and split == "train"
        self.include_silence_in_test = bool(include_silence_in_test)
        self.silence_test_samples = silence_eval_samples if silence_test_samples is None else int(silence_test_samples)
        self.unknown_keep_prob = unknown_keep_prob

        if not self.audio_root.exists():
            raise FileNotFoundError(f"Audio root does not exist: {self.audio_root}")

        self.class_to_idx = {word: i for i, word in enumerate(self.TARGET_WORDS)}
        self.class_to_idx["unknown"] = 10
        self.class_to_idx["silence"] = 11

        val_list_path = self.root_dir / "validation_list.txt"
        test_list_path = self.root_dir / "testing_list.txt"
        if not val_list_path.exists() or not test_list_path.exists():
            raise FileNotFoundError(
                f"Missing split files in {self.root_dir}. Expected validation_list.txt and testing_list.txt"
            )

        with open(val_list_path, "r", encoding="utf-8") as file_obj:
            val_files = set(file_obj.read().splitlines())
        with open(test_list_path, "r", encoding="utf-8") as file_obj:
            test_files = set(file_obj.read().splitlines())

        self.file_paths = []
        self.labels = []
        self.bg_noise_files = []

        for label_dir in self.audio_root.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            if label == "_background_noise_":
                self.bg_noise_files.extend(label_dir.glob("*.wav"))
                continue

            mapped_label = label if label in self.TARGET_WORDS else "unknown"
            label_idx = self.class_to_idx[mapped_label]

            for wav_path in label_dir.glob("*.wav"):
                rel_path = f"{label}/{wav_path.name}"
                is_val = rel_path in val_files
                is_test = rel_path in test_files

                if split == "val" and is_val:
                    self.file_paths.append(wav_path)
                    self.labels.append(label_idx)
                elif split == "test" and is_test:
                    self.file_paths.append(wav_path)
                    self.labels.append(label_idx)
                elif split == "train" and not is_val and not is_test:
                    if mapped_label == "unknown" and random.random() > self.unknown_keep_prob:
                        continue
                    self.file_paths.append(wav_path)
                    self.labels.append(label_idx)

        should_add_silence = split != "test" or self.include_silence_in_test
        if should_add_silence:
            if split == "train":
                silence_count = silence_train_samples
            elif split == "val":
                silence_count = silence_eval_samples
            else:
                silence_count = self.silence_test_samples
            for _ in range(silence_count):
                self.file_paths.append(self.SILENCE_MARKER)
                self.labels.append(self.class_to_idx["silence"])

        if not self.bg_noise_files and any(path == self.SILENCE_MARKER for path in self.file_paths):
            raise RuntimeError("No background noise files found in _background_noise_ for silence generation")

        self.mel_spec = T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=n_mels)
        self.amp_to_db = T.AmplitudeToDB()
        if self.apply_augment:
            self.spec_aug = torch.nn.Sequential(
                T.TimeMasking(time_mask_param=time_mask),
                T.FrequencyMasking(freq_mask_param=freq_mask),
            )

    def _to_mono(self, waveform):
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)

    def _load_waveform(self, path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WavFileWarning)
            sample_rate, data = wavfile.read(str(path))

        if data.dtype == np.int16:
            waveform = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            waveform = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            waveform = (data.astype(np.float32) - 128.0) / 128.0
        else:
            waveform = data.astype(np.float32)

        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        return waveform, int(sample_rate)

    def _resample_if_needed(self, waveform, sample_rate):
        if sample_rate == self.sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

    def _pad_or_truncate(self, waveform, target_length=16000):
        current_length = waveform.shape[1]
        if current_length < target_length:
            return torch.nn.functional.pad(waveform, (0, target_length - current_length))
        return waveform[:, :target_length]

    def _get_random_silence(self):
        """Extract a random 1-second chunk from a background noise file."""
        noise_file = random.choice(self.bg_noise_files)
        waveform, sample_rate = self._load_waveform(noise_file)
        waveform = self._to_mono(waveform)
        waveform = self._resample_if_needed(waveform, sample_rate)
        if waveform.shape[1] < 16000:
            return self._pad_or_truncate(waveform, target_length=16000)

        max_start = waveform.shape[1] - 16000
        start = random.randint(0, max_start)
        return waveform[:, start:start + 16000]

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        if path == self.SILENCE_MARKER:
            waveform = self._get_random_silence()
        else:
            waveform, sample_rate = self._load_waveform(path)
            waveform = self._to_mono(waveform)
            waveform = self._resample_if_needed(waveform, sample_rate)
            waveform = self._pad_or_truncate(waveform, target_length=16000)

        spec = self.amp_to_db(self.mel_spec(waveform))
        if self.apply_augment:
            spec = self.spec_aug(spec)

        return spec, label

    def __len__(self):
        return len(self.file_paths)