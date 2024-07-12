import os
import csv
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

from audio_tokenizer import AudioTokenizer
from tts_processor import TTSProcessor

warnings.filterwarnings("ignore")


@torch.no_grad()
def process_text(
    subset: Dataset,
    device: str,
    process_id: int,
    queue: Queue,
    proccesed_count: Value,
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
    for text, index in zip(batch_prompt, batch_index):
        for attempt in range(max_retries):
            try:
                audio = tts_processor.convert_text_to_audio(text).cpu().numpy()
                audio_tokens = audio_tokenizer.process_single_audio(
                    (audio, sample_rate)
                )
                queue.put(
                    {
                        "index": index,
                        "audio": audio,
                        "tokens": audio_tokens,
                        "sample_rate": sample_rate,
                    }
                )
                with proccesed_count.get_lock():  # Increment the shared value with a lock
                    proccesed_count.value += 1

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for index {index}: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for index {index}")


def save_batch(batch, audio_dir: str, csv_writer, save_count: Value):
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
        logger.info(f"Saving index {index} to {audio_dir}.")

        audio_path = os.path.join(audio_dir, f"audio_{index}.wav")

        torchaudio.save(audio_path, audio, 24_000)
        csv_writer.writerow([index, tokens])
        with save_count.get_lock():  # Increment the shared value with a lock
            save_count.value += 1


def save_audio_tokens(
    queue: Queue,
    audio_dir: str,
    token_saved_path: str,
    save_count: Value,
    proccesed_count: Value,
    queue_timeout: int = 5,
    batch_size: int = 10,
):
    """Save audio and tokens to disk.

    Args:
        queue (Queue): The queue containing the audio tokens.
        audio_dir (str): The directory to save the audio.byobu
        token_dir (str): The directory to save the tokens.
        saved_count (Value): The shared value to keep track of the number of saved examples.
    """
    with open(token_saved_path, "w", newline="") as file:
        batch = []
        csv_writer = csv.writer(
            file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL, escapechar="\\"
        )
        # Write the header
        csv_writer.writerow(["index", "tokens"])
        while True:
            time.sleep(queue_timeout)
            try:
                audio_token = queue.get()
                if audio_token is None:
                    break

                batch.append(audio_token)
                if len(batch) >= batch_size:
                    save_batch(batch, audio_dir, csv_writer, save_count)
                    batch = []

            except Empty:
                if batch:
                    save_batch(batch, audio_dir, csv_writer, save_count)
                    batch = []
            logger.info(f"Saved {save_count.value} examples.")
            logger.info(f"Current queue size: {queue.qsize()}.")
            logger.info(f"Current processed count: {proccesed_count.value}.")


def run_pipeline(
    dataset: Dataset,
    devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    num_procs_per_device: int = 14,
    audio_dir: str = "./new_audio_v3",
    token_saved_path: str = "./new_tokens_v3.csv",
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

    num_workers = len(devices) * num_procs_per_device
    logger.info(f"Using devices: {devices}")
    logger.info(f"Running pipeline with {num_workers} workers.")
    logger.info(f"New audio will be saved to {audio_dir}.")
    logger.info(f"New tokens will be saved to {token_saved_path}.")

    # Split the dataset into chunks
    chunk_size = ceil(len(dataset) / num_workers)
    chunks = [
        dataset.select(range(i, min(i + chunk_size, len(dataset))))
        for i in range(0, len(dataset), chunk_size)
    ]

    # Start the processes
    queue = Queue()
    save_count = Value("i", 0)
    proccesed_count = Value("i", 0)
    save_process = Process(
        target=save_audio_tokens,
        args=(queue, audio_dir, token_saved_path, save_count, proccesed_count),
    )
    save_process.start()

    worker_processes = []
    for i, chunk in enumerate(chunks):
        device = devices[i % len(devices)]
        p = Process(
            target=process_text, args=(chunk, device, i, queue, proccesed_count)
        )
        p.start()
        worker_processes.append(p)

    for p in worker_processes:
        p.join()

    queue.put(None)
    save_process.join()


if __name__ == "__main__":
    DO_TEST = False
    with open(
        "/home/root/Workspace/synthetic_data_generation/sound_instruct_llama3/data/remaining_indices.json"
    ) as f:
        remaining_indices = json.load(f)

    dataset = load_dataset("jan-hq/instruction-speech-v1.5", split="train")

    if DO_TEST:
        logger.info("Running test pipeline with 50 examples.")
        run_pipeline(
            dataset=dataset.select(range(50)),
            devices=["cuda:0", "cuda:1"],
            num_procs_per_device=2,
            audio_dir="./new_audio_test",
            token_saved_path="./new_tokens_test.csv",
        )
    else:
        logger.info(f"Continue process for {len(remaining_indices)} examples")
        run_pipeline(dataset.select(remaining_indices))
