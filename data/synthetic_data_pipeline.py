import os
import json
from typing import List, Dict, Any
import asyncio

import torch
import torchaudio
from datasets import Dataset
from loguru import logger as logging
from tqdm.asyncio import tqdm

from audio_tokenizer import AudioTokenizer
from tts_processor import TTSProcessor


class CombinedTTSTokenizer:
    def __init__(
        self,
        device: str = "cuda:0",
        sample_rate: int = 24_000,
        max_retries: int = 3,
    ):
        """Initialize the processor with TTS and audio tokenizer models."""
        self.tts_processor = TTSProcessor(device=device)
        self.audio_tokenizer = AudioTokenizer(device=device)
        self.device = device
        self.sample_rate = sample_rate
        self.max_retries = max_retries

    @torch.no_grad()
    def process_text(self, text: str, index: int) -> Dict[str, Any]:
        """Process text to generate audio and tokens in a single call.
        
        Args:
            text: The text to process.
            index: The index of the text in the dataset.

        Returns:
            A dictionary containing the index, audio, and tokens if successful,
            otherwise None.
        """
        for attempt in range(self.max_retries):
            try:
                # Generate audio from text
                audio = self.tts_processor.convert_text_to_audio(text)

                # Generate tokens from audio
                audio_tokens = self.audio_tokenizer.process_single_audio(
                    (audio, self.sample_rate)
                )

                return {"index": index, "audio": audio, "tokens": audio_tokens}
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt + 1} failed for index {index}: {str(e)}"
                )
                if attempt == self.max_retries - 1:
                    logging.error(f"All attempts failed for index {index}")
                    return None
        return None


async def save_results(result: Dict[str, Any], audio_dir: str, token_dir: str):
    """Asynchronously save audio and tokens to files.
    
    Args:
        result: A dictionary containing the index, audio, and tokens.
        audio_dir: The directory to save audio files.
        token_dir: The directory to save token files.
    """
    if result is None:
        return

    index = result["index"]
    audio = result["audio"]
    tokens = result["tokens"]

    # Save audio
    audio_path = os.path.join(audio_dir, f"audio_{index}.wav")
    await asyncio.to_thread(torchaudio.save, audio_path, audio, 24000)

    # Save tokens
    token_path = os.path.join(token_dir, f"audio_token_{index}.json")
    await asyncio.to_thread(lambda: json.dump(tokens, open(token_path, "w")))


async def save_failed_indices(failed_indices: List[int], failed_indices_file: str):
    """Save the indices of failed processing attempts to a file.
    
    Args:
        failed_indices: A list of indices that failed to process.
        failed_indices_file: The file to save the failed indices.
    """
    await asyncio.to_thread(lambda: json.dump(failed_indices, open(failed_indices_file, "w")))


async def process_and_save(
    processor: CombinedTTSTokenizer,
    item: Dict[str, Any],
    audio_dir: str,
    token_dir: str,
    failed_indices: List[int],
):
    """Process text and save results asynchronously.
    
    Args:
        processor: The processor to use for the text.
        item: A dictionary containing the text and index.
        audio_dir: The directory to save audio files.
        token_dir: The directory to save token files.
        failed_indices: A list to collect indices of failed processing attempts.
    """
    result = await asyncio.to_thread(
        processor.process_text, item["prompt"], item["index"]
    )
    if result:
        await save_results(result, audio_dir, token_dir)
    else:
        failed_indices.append(item["index"])


async def process_batch(
    processors: List[CombinedTTSTokenizer],
    batch: Dataset,
    audio_dir: str,
    token_dir: str,
    pbar: tqdm,
    failed_indices: List[int],
):
    """Process a batch of text inputs and save results.
    
    Args:
        processors: A list of processors to use for the batch.
        batch: A batch of text inputs.
        audio_dir: The directory to save audio files.
        token_dir: The directory to save token files.
        pbar: A tqdm progress bar to update.
        failed_indices: A list to collect indices of failed processing attempts.
    """
    tasks = []
    for item, processor in zip(batch, processors):
        task = asyncio.create_task(
            process_and_save(processor, item, audio_dir, token_dir, failed_indices)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)
    pbar.update(len(batch))


async def run_multi_works(
    gpu_ids: List[int],
    dataset: Dataset,
    audio_dir: str = "./new_audio",
    token_dir: str = "./new_tokens",
    failed_indices_file: str = "./failed_indices.json",
    num_procs_per_gpu: int = 7,
):
    """Run multiple workers to process a dataset across multiple GPUs.
    
    Args:
        gpu_ids: A list of GPU IDs to use for processing.
        dataset: The dataset to process.
        audio_dir: The directory to save audio files.
        token_dir: The directory to save token files.
        failed_indices_file: The file to save the failed indices.
        num_procs_per_gpu: The number of processors to use per GPU.
    """
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(token_dir, exist_ok=True)

    processors = [
        CombinedTTSTokenizer(device=f"cuda:{gpu_id}")
        for gpu_id in gpu_ids
        for _ in range(num_procs_per_gpu)
    ]
    total_samples = len(dataset)
    total_processors = len(processors)
    failed_indices = []

    async with tqdm(total=total_samples, desc="Processing samples") as pbar:
        for i in range(0, total_samples, total_processors):
            batch = dataset.select(range(i, i + total_processors))
            await process_batch(
                processors[: len(batch)],  # Use only as many processors as needed
                batch,
                audio_dir,
                token_dir,
                pbar,
                failed_indices,
            )

    await save_failed_indices(failed_indices, failed_indices_file)
    logging.info("Processing completed.")
    logging.info(f"Failed indices saved to {failed_indices_file}")


if __name__ == "__main__":
    gpu_ids = [0, 1]  # Adjust based on available GPUs
    dataset = Dataset.from_dict(
        {"prompt": ["Hello world"] * 1000, "index": list(range(1000))}
    )

    asyncio.run(run_multi_works(gpu_ids, dataset))