from typing import List, Optional

import torch
import torch.nn.functional as F
import torchaudio
import whisper
from datasets import load_dataset
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm


class WeightedDataset(Dataset):
    """Wrapper for dataset with weight"""

    def __init__(self, dataset: Dataset, weight: float = 1.0):
        self.dataset = dataset
        self.weight = weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.weight


class WhisperDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        txt_label: str = "transcription",
        language: str = "vi",
        num_samples: Optional[int] = None,
        task: str = "train",
        max_tokens: int = 200,
    ):
        self.split = split

        if "libritts_r_filtered" in dataset_dir:
            if split == "validation":
                self.dataset = load_dataset(dataset_dir, "clean", split="dev.clean")
            elif split == "test":
                self.dataset = load_dataset(dataset_dir, "clean", split="test.clean")
            else:
                self.dataset = load_dataset(
                    dataset_dir, "clean", split="train.clean.360"
                )

            self.dataset = self.dataset.select_columns(["audio", "text_normalized"])
            self.dataset = self.dataset.rename_column(
                "text_normalized", "transcription"
            )
            if rank_zero_only.rank == 0:
                print(f"🚀 Loaded {len(self.dataset)} samples from {dataset_dir}")

        # TODO: load vivoice from jan-hq directly
        elif "viVoice" in dataset_dir:
            full_dataset = load_dataset(dataset_dir)
            full_data = full_dataset["train"]
            shuffled_data = full_data.shuffle(seed=42)

            total_size = len(shuffled_data)
            test_size = 10000
            val_size = 10000
            train_size = total_size - val_size - test_size

            # === Dataset Statistics ===
            # Dataset 0:
            # - Size: 867,772 samples
            # - Weight: 0.1
            # - Effective sampling ratio: 46.2%
            # Dataset 1:
            # - Size: 112,326 samples
            # - Weight: 0.9
            # - Effective sampling ratio: 53.8%
            # Total samples available: 980,098

            if split == "train":
                self.dataset = shuffled_data.select(range(train_size))
            elif split == "validation":
                self.dataset = shuffled_data.select(
                    range(train_size, train_size + val_size)
                )
            elif split == "test":
                self.dataset = shuffled_data.select(
                    range(train_size + val_size, total_size)
                )
            self.dataset = self.dataset.select_columns(["audio", "text"])
            self.dataset = self.dataset.rename_column("text", "transcription")
            if rank_zero_only.rank == 0:
                print(
                    f"🚀 Split {dataset_dir} into {len(self.dataset)} samples for {split}"
                )
        else:
            self.dataset = load_dataset(dataset_dir, split=split)
            if rank_zero_only.rank == 0:
                print(f"🚀 Loaded {len(self.dataset)} samples from {dataset_dir}")

        if num_samples:
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

        self.txt_label = txt_label
        self.language = language
        self.task = task
        self.max_audio_length = 30 * 16000  # 30 seconds at 16kHz
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=language, task="transcribe"
        )

        self.max_tokens = max_tokens


    def _get_audio_duration(self, example: dict) -> float:
        return len(example["audio"]["array"]) / example["audio"]["sampling_rate"]

    def __len__(self):
        return len(self.dataset)

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if len(audio) > self.max_audio_length:
            return audio[: self.max_audio_length]
        return F.pad(audio, (0, self.max_audio_length - len(audio)), value=0)

    def __getitem__(self, idx):
        if self.split == "test":
            example = self.dataset[idx]

            samples = torch.tensor(example["audio"]["array"], dtype=torch.float32)

            if example["audio"]["sampling_rate"] != 16000:
                resampler = torchaudio.transforms.Resample(
                    example["audio"]["sampling_rate"], 16000
                )
                samples = resampler(samples)

            # Normalize audio
            if samples.abs().max() > 0:
                samples = samples / samples.abs().max()

            # Pad audio
            samples = self.pad_audio(samples)

            if samples.abs().max() > 0:
                samples = samples / samples.abs().max()

            # Process text tokens
            tokens = list(
                self.tokenizer.sot_sequence_including_notimestamps
            ) + self.tokenizer.encode(example[self.txt_label])

            # Pad tokens
            rpad = self.max_tokens - len(tokens)
            output_toks = F.pad(
                torch.tensor(tokens, dtype=torch.long),
                (0, rpad),
                value=self.tokenizer.eot,
            )

            return samples, output_toks

        # Get single sample
        example = self.dataset[idx]

        # Process audio
        samples = torch.tensor(example["audio"]["array"], dtype=torch.float32)
        if samples.dim() == 2:
            samples = samples.mean(0)

        # Resample if needed
        if example["audio"]["sampling_rate"] != 16000:
            resampler = torchaudio.transforms.Resample(
                example["audio"]["sampling_rate"], 16000
            )
            samples = resampler(samples)

        # Normalize audio
        if samples.abs().max() > 0:
            samples = samples / samples.abs().max()

        # Pad audio
        samples = self.pad_audio(samples)

        # Create mask for attention
        mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
        audio_frames = min(len(samples), self.max_audio_length) // 320
        mask[:audio_frames] = 1

        # Process text tokens
        tokens = list(
            self.tokenizer.sot_sequence_including_notimestamps
        ) + self.tokenizer.encode(example[self.txt_label])

        # Pad tokens
        rpad = self.max_tokens - len(tokens)

        in_ttoks = F.pad(
            torch.tensor(tokens, dtype=torch.long),
            (0, rpad),
            value=self.tokenizer.eot,
        )
        out_ttoks = F.pad(
            torch.tensor(tokens[1:] + [self.tokenizer.eot], dtype=torch.long),
            (0, rpad),
            value=-100,
        )

        return samples, mask, in_ttoks, out_ttoks

        # Get the group of samples to concatenate
        group = self.grouped_indices[idx]

        # Process and concatenate audio
        audio_samples = []
        texts = []
        total_frames = 0

        for sample_idx in group:
            example = self.dataset[sample_idx]
            # Process audio
            samples = torch.tensor(example["audio"]["array"], dtype=torch.float32)
            if samples.dim() == 2:
                samples = samples.mean(0)

            # Resample if needed
            if example["audio"]["sampling_rate"] != 16000:
                resampler = torchaudio.transforms.Resample(
                    example["audio"]["sampling_rate"], 16000
                )
                samples = resampler(samples)

            # Normalize audio
            if samples.abs().max() > 0:
                samples = samples / samples.abs().max()

            audio_samples.append(samples)
            texts.append(example[self.txt_label])
            total_frames += len(samples)

        # Concatenate audio samples
        concatenated_audio = torch.cat(audio_samples)
        concatenated_text = " ".join(texts)

        # Pad if necessary
        concatenated_audio = self.pad_audio(concatenated_audio)

        # Create mask for attention
        mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
        audio_frames = min(len(concatenated_audio), self.max_audio_length) // 320
        mask[:audio_frames] = 1

        # Process text tokens
        tokens = list(
            self.tokenizer.sot_sequence_including_notimestamps
        ) + self.tokenizer.encode(concatenated_text)

        # Pad tokens
        rpad = self.max_tokens - len(tokens)

        in_ttoks = F.pad(
            torch.tensor(tokens, dtype=torch.long),
            (0, rpad),
            value=self.tokenizer.eot,
        )
        out_ttoks = F.pad(
            torch.tensor(tokens[1:] + [self.tokenizer.eot], dtype=torch.long),
            (0, rpad),
            value=-100,
        )

        return concatenated_audio, mask, in_ttoks, out_ttoks

    def _print_stats_summary(self):
        """Print summary statistics and save to file"""
        if not self.stats:
            return

        print("\n=== Dataset Statistics ===")
        print(f"Total groups: {len(self.grouped_indices)}")

        # Calculate averages
        avg_samples = sum(s["num_samples"] for s in self.stats) / len(self.stats)
        avg_frames = sum(s["total_frames"] for s in self.stats) / len(self.stats)
        avg_tokens = sum(s["token_length"] for s in self.stats) / len(self.stats)

        print(f"Average samples per group: {avg_samples:.2f}")
        print(f"Average frames per group: {avg_frames:.2f}")
        print(f"Average tokens per group: {avg_tokens:.2f}")

        # Save to file
        with open("dataset_stats.txt", "w") as f:
            f.write("Group Index,Num Samples,Total Frames,Text Length,Token Length\n")
            for stat in self.stats:
                f.write(
                    f"{stat['group_idx']},{stat['num_samples']},"
                    f"{stat['total_frames']},{stat['text_length']},"
                    f"{stat['token_length']}\n"
                )
        print(f"Statistics saved to dataset_stats.txt")


def load_whisper_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    language: str = "vi",
    validation: bool = False,
    num_samples: Optional[int] = None,
    weight: float = 1.0,
    max_tokens: int = 200,
) -> WeightedDataset:
    """Load dataset with weight"""
    split = "validation" if validation else "train"

    dataset = WhisperDataset(
        dataset_dir=dataset_dir,
        split=split,
        txt_label=txt_label,
        language=language,
        num_samples=num_samples,
        max_tokens=max_tokens,
    )
    return WeightedDataset(dataset, weight)


def load_multiple_datasets(
    dataset_configs: List[dict],
    validation: bool = False,
) -> ConcatDataset:
    """Load multiple datasets with their weights"""
    datasets = []
    for config in dataset_configs:
        dataset = load_whisper_dataset(validation=validation, **config)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def load_test_dataset(
    dataset_dir: str,
    txt_label: str = "transcription",
    language: str = "vi",
    num_samples: Optional[int] = None,
) -> WhisperDataset:
    return WhisperDataset(
        dataset_dir=dataset_dir,
        split="test",
        txt_label=txt_label,
        language=language,
        num_samples=num_samples,
    )
