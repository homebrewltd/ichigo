# TODO: fix transcription column name
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchaudio
import whisper
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from tqdm import tqdm


def process_audio(audio, sr=16000, max_length=30 * 16000):
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio, dtype=torch.float32)

    if audio.dim() == 2:
        audio = audio.mean(0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    if audio.abs().max() > 0:
        audio = audio / audio.abs().max()

    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = F.pad(audio, (0, max_length - len(audio)), value=0)

    return audio


def group_samples(dataset: Dataset, max_duration: float = 30.0) -> List[List[int]]:
    """Group samples to maximize usage of max_duration-second window"""
    groups = []
    current_group = []
    current_duration = 0.0

    for idx in tqdm(range(len(dataset)), desc="Grouping samples", unit="sample"):
        duration = (
            len(dataset[idx]["audio"]["array"]) / dataset[idx]["audio"]["sampling_rate"]
        )

        if current_duration + duration <= max_duration:
            current_group.append(idx)
            current_duration += duration
        else:
            if current_group:
                groups.append(current_group)
            current_group = [idx]
            current_duration = duration

    if current_group:
        groups.append(current_group)

    print(f"✓ Created {len(groups)} groups from {len(dataset)} samples")
    return groups


def process_split(
    dataset: Dataset,
    tokenizer,
    split: str,
    language: str = "vi",
    txt_label: str = "transcription",
    max_tokens: int = 200,
    is_test: bool = False,
):
    processed_data = {
        "audio": [],
        "input_tokens": [],
        "output_tokens": [],
        "mask": [],
    }

    if language == "vi" and not is_test:
        groups = group_samples(dataset)

        for group in tqdm(groups, desc=f"Processing {split} split"):
            audio_samples = []
            texts = []

            for idx in group:
                example = dataset[idx]
                audio = process_audio(
                    example["audio"]["array"], example["audio"]["sampling_rate"]
                )
                audio_samples.append(audio)
                texts.append(example[txt_label])

            audio = torch.cat(audio_samples)
            text = " ".join(texts)

            mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
            audio_frames = min(len(audio), 30 * 16000) // 320
            mask[:audio_frames] = 1

            tokens = list(
                tokenizer.sot_sequence_including_notimestamps
            ) + tokenizer.encode(text)

            rpad = max_tokens - len(tokens)
            in_tokens = F.pad(
                torch.tensor(tokens, dtype=torch.long), (0, rpad), value=tokenizer.eot
            )
            out_tokens = F.pad(
                torch.tensor(tokens[1:] + [tokenizer.eot], dtype=torch.long),
                (0, rpad),
                value=-100,
            )

            processed_data["audio"].append(audio.numpy())
            processed_data["mask"].append(mask.numpy())
            processed_data["input_tokens"].append(in_tokens.numpy())
            processed_data["output_tokens"].append(out_tokens.numpy())

    else:
        for example in tqdm(dataset, desc=f"Processing {split} split"):
            audio = process_audio(
                example["audio"]["array"], example["audio"]["sampling_rate"]
            )

            tokens = list(
                tokenizer.sot_sequence_including_notimestamps
            ) + tokenizer.encode(example[txt_label])

            rpad = max_tokens - len(tokens)
            in_tokens = F.pad(
                torch.tensor(tokens, dtype=torch.long), (0, rpad), value=tokenizer.eot
            )

            processed_data["audio"].append(audio.numpy())
            processed_data["input_tokens"].append(in_tokens.numpy())

            if not is_test:
                mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
                audio_frames = min(len(audio), 30 * 16000) // 320
                mask[:audio_frames] = 1

                out_tokens = F.pad(
                    torch.tensor(tokens[1:] + [tokenizer.eot], dtype=torch.long),
                    (0, rpad),
                    value=-100,
                )

                processed_data["mask"].append(mask.numpy())
                processed_data["output_tokens"].append(out_tokens.numpy())

    if is_test:
        for _ in range(len(processed_data["audio"])):
            mask = torch.zeros(30 * 16000 // 320, dtype=torch.bool)
            processed_data["mask"].append(mask.numpy())
            processed_data["output_tokens"].append(
                torch.full((max_tokens,), -100, dtype=torch.long).numpy()
            )

    # Use the same features for all splits
    features = Features(
        {
            "audio": Sequence(Value("float32")),
            "mask": Sequence(Value("bool")),
            "input_tokens": Sequence(Value("int64")),
            "output_tokens": Sequence(Value("int64")),
        }
    )

    return Dataset.from_dict(processed_data, features=features)


def process_dataset(
    dataset_path: str,
    output_path: str,
    language: str = "vi",
    txt_label: str = "transcription",
    num_samples: Optional[int] = None,
    max_tokens: int = 200,
):
    tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=language, task="transcribe"
    )

    # Load datasets for all splits
    if "libritts_r_filtered" in dataset_path:
        train_dataset = load_dataset(dataset_path, "clean", split="train.clean.360")
        valid_dataset = load_dataset(dataset_path, "clean", split="dev.clean")
        test_dataset = load_dataset(dataset_path, "clean", split="test.clean")

        # Rename columns for consistency
        for ds in [train_dataset, valid_dataset, test_dataset]:
            ds = ds.select_columns(["audio", "text_normalized"])
            ds = ds.rename_column("text_normalized", "transcription")
    else:
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"]
        valid_dataset = dataset["validation"] if "validation" in dataset else None
        test_dataset = dataset["test"] if "test" in dataset else None

    if num_samples:
        train_dataset = train_dataset.select(
            range(min(num_samples, len(train_dataset)))
        )
        if valid_dataset:
            valid_dataset = valid_dataset.select(
                range(min(num_samples, len(valid_dataset)))
            )
        if test_dataset:
            test_dataset = test_dataset.select(
                range(min(num_samples, len(test_dataset)))
            )

    # Process each split
    processed_datasets = {}

    processed_datasets["train"] = process_split(
        train_dataset, tokenizer, "train", language, txt_label, max_tokens
    )

    if valid_dataset is not None:
        processed_datasets["validation"] = process_split(
            valid_dataset, tokenizer, "validation", language, txt_label, max_tokens
        )

    if test_dataset is not None:
        processed_datasets["test"] = process_split(
            test_dataset,
            tokenizer,
            "test",
            language,
            txt_label,
            max_tokens,
            is_test=True,
        )

    # Create DatasetDict and save
    dataset_dict = DatasetDict(processed_datasets)
    dataset_dict.save_to_disk(output_path)
    print(f"✅ Processed dataset saved to {output_path}")

    return dataset_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--language", type=str, default="vi")
    parser.add_argument("--txt_label", type=str, default="transcription")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    dataset_dict = process_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        language=args.language,
        txt_label=args.txt_label,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
    )

    if args.push_to_hub:
        print(f"🚀 Uploading to HuggingFace Hub: {args.push_to_hub}")
        dataset_dict.push_to_hub(args.push_to_hub)
        print("✅ Upload complete!")
