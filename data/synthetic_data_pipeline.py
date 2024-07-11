import os
import json
import warnings
import time
from math import ceil
from multiprocessing import Queue, Process, Value
from typing import List
from queue import Empty

import torch
import torchaudio
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm

from audio_tokenizer import AudioTokenizer
from tts_processor import TTSProcessor

warnings.filterwarnings("ignore")

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@torch.no_grad()
def process_text(
    subset: Dataset,
    device: str,
    process_id: int,
    queue: Queue,
    sample_rate: int = 24_000,
    max_retries: int = 3,
):
    """Convert text to audio and tokenize the audio.

    Args:
        example (Dataset): The dataset containing the text to convert.
        device (str, optional): The device to use for processing. Defaults to "cuda:0".
        sample_rate (int, optional): The sample rate of the audio. Defaults to 24_000.
        max_retries (int, optional): The maximum number of retries. Defaults to 3.

    Returns:
        dict: A dictionary containing the index, audio, and tokens or None if all attempts failed.
    """
    logger.info(f"Process {process_id} will process {len(subset)} examples.")
    batch_prompt = subset["prompt"]
    batch_index = subset["index"]
    tts_processor = TTSProcessor(device=device)
    audio_tokenizer = AudioTokenizer(device=device)
    for text, index in tqdm(
        zip(batch_prompt, batch_index),
        total=len(subset),
        desc=f"Process {process_id}",
        position=process_id,
    ):
        for attempt in range(max_retries):
            try:
                audio = tts_processor.convert_text_to_audio(text).cpu()
                audio_tokens = audio_tokenizer.process_single_audio((audio, sample_rate))
                queue.put({"index": index, "audio": audio, "tokens": audio_tokens})
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for index {index}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for index {index}")


def save_batch(batch, audio_dir: str, token_dir: str, saved_count):
    """Save a batch of audio and tokens to disk.

    Args:
        batch (List[dict]): The batch containing the audio tokens.
        audio_dir (str): The directory to save the audio.
        token_dir (str): The directory to save the tokens.
    """
    for audio_token in batch:
        index = audio_token["index"]
        audio = audio_token["audio"]
        tokens = audio_token["tokens"]
        logger.info(f"Saving index {index} to {audio_dir} and {token_dir}")

        audio_path = os.path.join(audio_dir, f"audio_{index}.wav")
        token_path = os.path.join(token_dir, f"{index}.pt")

        torchaudio.save(audio_path, audio, 24_000)
        torch.save(tokens, token_path)
    saved_count.value += len(batch)


def save_audio_tokens(
    queue: Queue,
    audio_dir: str,
    token_dir: str,
    saved_count: Value,
    queue_timeout: int = 5,
    batch_size: int = 10,
):
    """Save audio and tokens to disk.

    Args:
        queue (Queue): The queue containing the audio tokens.
        audio_dir (str): The directory to save the audio.
        token_dir (str): The directory to save the tokens.
        saved_count (Value): The shared value to keep track of the number of saved examples.
    """
    batch = []
    while True:
        logger.info(f"Saved {saved_count.value} examples.")
        time.sleep(queue_timeout)
        try:
            audio_token = queue.get()
            if audio_token is None:
                break

            batch.append(audio_token)
            if len(batch) >= batch_size:
                save_batch(batch, audio_dir, token_dir)
                batch = []

        except Empty:
            if batch:
                save_batch(batch, audio_dir, token_dir)
                batch = []


def run_pipeline(
    dataset: Dataset,
    devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    num_procs_per_device: int = 10,
    audio_dir: str = "./new_audio",
    token_dir: str = "./new_tokens",
):
    """Run the pipeline to convert text to audio and tokenize the audio.

    Args:
        dataset (Dataset): The dataset to process.
        devices (List[str], optional): The devices to use for processing. Defaults to ["cuda:0"].
        num_procs_per_device (int, optional): The number of processes per device. Defaults to 2.
        audio_dir (str, optional): The directory to save the audio. Defaults to "./new_audio".
        token_dir (str, optional): The directory to save the tokens. Defaults to "./new_tokens".
    """
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(token_dir, exist_ok=True)

    num_workers = len(devices) * num_procs_per_device
    logger.info(f"Using devices: {devices}")
    logger.info(f"Running pipeline with {num_workers} workers.")
    logger.info(f"New audio will be saved to {audio_dir}.")
    logger.info(f"New tokens will be saved to {token_dir}.")

    chunk_size = ceil(len(dataset) / num_workers)
    chunks = [dataset.select(range(i, min(i + chunk_size, len(dataset)))) for i in range(0, len(dataset), chunk_size)]

    queue = Queue()
    saved_count = Value("i", 0)
    save_process = Process(target=save_audio_tokens, args=(queue, audio_dir, token_dir, saved_count))
    save_process.start()

    worker_processes = []
    for i, chunk in enumerate(chunks):
        device = devices[i % len(devices)]
        p = Process(target=process_text, args=(chunk, device, i, queue))
        p.start()
        worker_processes.append(p)

    for p in worker_processes:
        p.join()

    queue.put(None)
    save_process.join()


if __name__ == "__main__":
    with open("/home/root/Workspace/synthetic_data_generation/sound_instruct_llama3/data/remaining_indices.json") as f:
        remaining_indices = json.load(f)

    logger.info(f"Continue process for {len(remaining_indices)} examples")
    dataset = load_dataset("jan-hq/instruction-speech-v1.5", split="train")
    run_pipeline(dataset.select(remaining_indices))