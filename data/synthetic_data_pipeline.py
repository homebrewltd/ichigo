"""Pipeline to convert text to audio and tokenize the audio.

This script allows multiple processes to convert text to
audio and tokenize the audio, push the output to a queue
for a different process to save the audio and tokens to disk.
The pipeline is designed to handle large datasets and can be
run on multiple GPUs, each GPU running multiple processes."""

import json
import time
import warnings
import sys
from math import ceil
from multiprocessing import Queue, Process, Value
from typing import List
from queue import Empty

import torch
import pyarrow as pa
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
    processed_count: Value,
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
                audio = tts_processor.convert_text_to_audio(text)  # torch tensor
                audio_tokens = audio_tokenizer.process_single_audio(
                    (audio, sample_rate)
                )  # list(int64)
                queue.put(
                    {
                        "index": index,
                        "audio": audio.cpu()
                        .squeeze(0)
                        .numpy(),  # flatten the tensor for saving to arrow array
                        "tokens": audio_tokens,
                    }
                )
                with processed_count.get_lock():  # Increment the shared value with a lock
                    processed_count.value += 1

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for index {index}: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for index {index}")


def save_batch(batch, writer, save_count: Value):
    """Save a batch of audio and tokens to disk.

    Args:
        batch (List[dict]): The batch containing the audio tokens.
        audio_dir (str): The directory to save the audio.
        token_dir (str): The directory to save the tokens.
    """
    # Gather the data
    index = [audio_token["index"] for audio_token in batch]
    audio = [audio_token["audio"] for audio_token in batch]
    tokens = [audio_token["tokens"] for audio_token in batch]
    # sample_rate = [audio_token["sample_rate"] for audio_token in batch]

    # Create pa.array from the data
    index_array = pa.array(index, type=pa.int64())
    audio_array = pa.array(audio, type=pa.list_(pa.float32()))
    tokens_array = pa.array(tokens, type=pa.list_(pa.int64()))
    # sample_rate_array = pa.array(sample_rate, type=pa.int64())

    # Create batch
    batch = pa.record_batch(
        [index_array, audio_array, tokens_array],
        names=["index", "audio", "tokens"],
    )

    # Write the batch to the file
    writer.write_batch(batch)
    with save_count.get_lock():  # Increment the shared value with a lock
        save_count.value += len(batch)


def signal_handler(signum, frame):
    """Handle the termination signal.

    Gracefully close the writer and exit the process when
    the termination signal is received.

    Args:
        signum (int): The signal number.
        frame (frame): The frame object.
    """
    global writer, batch, save_count
    print("Received termination signal. Performing cleanup...")

    if batch:
        save_batch(batch, writer, save_count)

    if writer:
        writer.close()

    print("Cleanup completed. Exiting.")
    sys.exit(0)


def save_audio_tokens(
    queue: Queue,
    saved_path: str,
    save_count: Value,
    processed_count: Value,
    queue_timeout: int = 60,
    batch_size: int = 100,
):
    """Save audio and tokens to disk.

    Args:
        queue (Queue): The queue containing the audio tokens.
        audio_dir (str): The directory to save the audio.byobu
        token_dir (str): The directory to save the tokens.
        saved_count (Value): The shared value to keep track of the number of saved examples.
    """
    # Time sleep due to long model init time
    global writer, batch  # Declare the global variables for the signal handler
    time.sleep(300)

    # Prepare pa schema
    schema = pa.schema(
        [
            pa.field("index", pa.int64()),  # int64
            pa.field("audio", pa.list_(pa.float32())),  # tensor
            pa.field("tokens", pa.list_(pa.int64())),  # list(int64)
            # pa.field("sample_rate", pa.int64()),  # int64
        ]
    )
    with pa.OSFile(saved_path, "wb") as sink:
        writer = pa.ipc.new_file(sink, schema)
        batch = []
        try:
            while True:
                try:
                    audio_token = queue.get(timeout=queue_timeout)
                    if audio_token is None:
                        break

                    batch.append(audio_token)
                    if len(batch) >= batch_size:
                        save_batch(batch, writer, save_count)
                        batch = []

                except Empty:
                    if batch:
                        save_batch(batch, writer, save_count)
                        batch = []

                # Log the progress
                logger.info(f"Saved {save_count.value} examples.")
                logger.info(f"Current queue size: {queue.qsize()}.")
                logger.info(f"Current processed count: {processed_count.value}.")
        finally:
            if batch:
                save_batch(batch, writer, save_count)
            writer.close()


def run_pipeline(
    dataset: Dataset,
    devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    num_procs_per_device: int = 14,
    saved_path: str = "./new_tokens_v4.arrow",
):
    """Run the pipeline to convert text to audio and tokenize the audio.

    Args:
        dataset (Dataset): The dataset to process.
        devices (List[str], optional): The devices to use for processing. Defaults to ["cuda:0"].
        num_procs_per_device (int, optional): The number of processes per device. Defaults to 2.
        audio_dir (str, optional): The directory to save the audio. Defaults to "./new_audio".
        token_dir (str, optional): The directory to save the tokens. Defaults to "./new_tokens".
    """
    num_workers = len(devices) * num_procs_per_device
    logger.info(f"Using devices: {devices}")
    logger.info(f"Running pipeline with {num_workers} workers.")
    logger.info(f"Data will be saved to {saved_path}.")

    # Split the dataset into chunks
    chunk_size = ceil(len(dataset) / num_workers)
    chunks = [
        dataset.select(range(i, min(i + chunk_size, len(dataset))))
        for i in range(0, len(dataset), chunk_size)
    ]

    # Start the processes
    queue = Queue()
    save_count = Value("i", 0)
    processed_count = Value("i", 0)
    save_process = Process(
        target=save_audio_tokens,
        args=(queue, saved_path, save_count, processed_count),
    )
    save_process.start()

    worker_processes = []
    for i, chunk in enumerate(chunks):
        device = devices[i % len(devices)]
        p = Process(
            target=process_text, args=(chunk, device, i, queue, processed_count)
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
        "/home/root/Workspace/synthetic_data_generation/sound_instruct_llama3/data/turn_4_processed.json"
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
            saved_path="./new_tokens_test.arrow",
        )
    else:
        logger.info(f"Continue process for {len(remaining_indices)} examples")
        run_pipeline(dataset.select(remaining_indices))
