import os
import json
import logging
import warnings
import time
from math import ceil
from multiprocessing import Process, Value
from typing import List

import torch
import pyarrow as pa
import pyarrow.csv as pa_csv
from datasets import Dataset, load_dataset

from audio_tokenizer import AudioTokenizer
from tts_processor import TTSProcessor

warnings.filterwarnings("ignore")


# Logging configuration
def configure_logging(log_file=None):
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)

    # Create a logger
    logger = logging.getLogger(__name__)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    return logger


logger = configure_logging()


class CSVWriter:
    def __init__(self, file_path, schema):
        self.file_path = file_path
        self.schema = schema
        self.writer = None
        self.open()

    def open(self):
        self.writer = pa_csv.CSVWriter(self.file_path, self.schema)

    def write(self, batch):
        self.writer.write(batch)

    def close(self):
        if self.writer:
            self.writer.close()


def save_failed_indices(batch_of_failed_indices: List, file_path: str):
    with open(file_path, "w+") as f:
        f.write("\n".join(map(str, batch_of_failed_indices)) + "\n")


@torch.no_grad()
def process_and_save_text(
    subset: Dataset,
    device: str,
    process_id: int,
    processed_count: Value,
    save_dir: str,
    save_batch_size: int = 10,
    sample_rate: int = 24_000,
    max_retries: int = 3,
):
    """Process the text and save the audio tokens to a CSV file."""
    logger.info("Process %s will process %s examples.", process_id, len(subset))
    batch_prompt = subset["prompt"]
    batch_index = subset["index"]
    tts_processor = TTSProcessor(device=device)
    audio_tokenizer = AudioTokenizer(device=device)

    # Create a CSV writer for this process
    schema = pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("audio", pa.string()),
            pa.field("tokens", pa.string()),
        ]
    )
    csv_file_path = os.path.join(save_dir, f"audio_tokens_{process_id}.csv")
    csv_writer = CSVWriter(csv_file_path, schema)
    logger.info("Process %s will save to %s.", process_id, csv_file_path)

    failed_indices = []
    saved_failed_indice_path = os.path.join(
        save_dir, f"failed_indices_{process_id}.json"
    )
    logger.info(
        "Process %s will save failed indices to %s.",
        process_id,
        saved_failed_indice_path,
    )

    batch = []
    for text, index in zip(batch_prompt, batch_index):
        logger.info("Process %s processing item sample %s.", process_id, index)
        for attempt in range(max_retries):
            try:
                audio = tts_processor.convert_text_to_audio(text)
                audio_tokens = audio_tokenizer.process_single_audio(
                    (audio, sample_rate)
                )
                batch.append(
                    {
                        "index": index,
                        "audio": json.dumps(audio.cpu().squeeze(0).numpy().tolist()),
                        "tokens": json.dumps(audio_tokens),
                    }
                )

                if len(batch) >= save_batch_size:
                    save_batch(batch, csv_writer)
                    batch = []
                    save_failed_indices(failed_indices, saved_failed_indice_path)
                    failed_indices = []

                with processed_count.get_lock():
                    processed_count.value += 1
                break
            except Exception as e:
                logger.warning(
                    "Attempt %s failed for index %s: %s", attempt + 1, index, str(e)
                )
                if attempt == max_retries - 1:
                    logger.error("All attempts failed for index %s", index)
                    failed_indices.append(index)

    # Save any remaining items in the batch
    if batch:
        save_batch(batch, csv_writer)
    if failed_indices:
        save_failed_indices(failed_indices, saved_failed_indice_path)

    csv_writer.close()


def save_batch(batch, csv_writer):
    logger.info("Saving progress.")
    # Create pa.array from the data
    index_array = pa.array([item["index"] for item in batch], type=pa.int64())
    audio_array = pa.array([item["audio"] for item in batch], type=pa.string())
    tokens_array = pa.array([item["tokens"] for item in batch], type=pa.string())

    # Create batch table
    batch_table = pa.Table.from_arrays(
        [index_array, audio_array, tokens_array], schema=csv_writer.schema
    )

    csv_writer.write(batch_table)


def create_non_overlapping_chunks(dataset, num_workers):
    indices = list(range(len(dataset)))
    chunk_size = ceil(len(indices) / num_workers)
    return [
        dataset.select(indices[i : i + chunk_size])
        for i in range(0, len(indices), chunk_size)
    ]


def run_pipeline(
    dataset,
    devices: List = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    num_procs_per_device: int = 14,
    save_dir: str = "./new_audio_v4_2",
    save_batch_size: int = 10,
):
    """Run the pipeline to convert text to audio and tokenize the audio."""
    os.makedirs(save_dir, exist_ok=True)
    num_workers = len(devices) * num_procs_per_device
    logger.info("Dataset size: %s", len(dataset))

    # Split the dataset into non-overlapping chunks
    chunks = create_non_overlapping_chunks(dataset, num_workers)
    total_items = sum(len(chunk) for chunk in chunks)
    logger.info("Total items in chunks: %s", total_items)

    processed_count = Value("i", 0)  # Value to store the number of items processed

    # Start the worker processes
    worker_processes = []
    for i, chunk in enumerate(chunks):
        device = devices[i % len(devices)]
        p = Process(
            target=process_and_save_text,
            args=(chunk, device, i, processed_count, save_dir, save_batch_size),
        )
        p.start()
        worker_processes.append(p)

    while any(p.is_alive() for p in worker_processes):
        # Log the progress every minute
        logger.info("Processed: %s", processed_count.value)
        time.sleep(60)

    # Wait for the worker processes to finish
    for p in worker_processes:
        p.join()

    logger.info("All worker processes have finished.")

    # Log the final counts
    logger.info("Final processed count: %s", processed_count.value)


if __name__ == "__main__":
    DO_TEST = False

    with open(
        "/home/root/Workspace/synthetic_data_generation/sound_instruct_llama3/data/remaining_indices.json"
    ) as f:
        remaining_indices = json.load(f)

    dataset = load_dataset("jan-hq/instruction-speech-v1.5", split="train")

    if DO_TEST:
        logger.info("Running test pipeline with 100 examples.")
        run_pipeline(
            dataset=dataset.select(range(100)),
            devices=["cuda:0", "cuda:1"],
            num_procs_per_device=2,
            save_dir="./new_test",
            save_batch_size=5,
        )
    else:
        logger.info("Continue process for %s examples", len(remaining_indices))
        run_pipeline(
            dataset=dataset.select(remaining_indices),
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            num_procs_per_device=14,
            save_dir="./new_audio_v4_2",
            save_batch_size=50,
        )
