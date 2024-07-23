"""Orchestrates the pipeline to convert text to audio and tokenize the audio
using multiple processes and GPU devices."""

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
import yaml
from datasets import Dataset, load_dataset

from audio_tokenizer import AudioTokenizer
from tts_processor import TTSProcessor

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    """Load the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def configure_logging(config: dict):
    """Configure the logging for the pipeline.

    Args:
        config (dict): The configuration dictionary.
    Returns:
        logging.Logger: The logger object
    """
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create a logger
    logger = logging.getLogger(__name__)

    # Set the logger's level to the lowest level you want to capture
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config["logging"]["console_level"])
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if config["logging"]["log_file"]:
        file_handler = logging.FileHandler(config["logging"]["log_file"])
        file_handler.setLevel(config["logging"]["file_level"])
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    return logger


class CSVWriter:
    def __init__(self, file_path, schema):
        """Initialize the CSV writer with the file path and schema.

        Args:
            file_path (str): The path to the CSV file.
            schema (pyarrow.Schema): The schema of the CSV file.
        """
        self.file_path = file_path
        self.schema = schema
        self.writer = None
        self.open()

    def open(self):
        """Open the CSV file for writing."""
        self.writer = pa_csv.CSVWriter(self.file_path, self.schema)

    def write(self, batch):
        """Write a batch of data to the CSV file."""
        self.writer.write(batch)

    def close(self):
        """Close the CSV file."""
        if self.writer:
            self.writer.close()


def save_failed_indices(batch_of_failed_indices: List, file_path: str):
    """Save the failed indices to a file."""
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
    """Process the text and save the audio tokens to a CSV file.

    Args:
        subset (Dataset): The subset of the dataset to process.
        device (str): The device to use for processing.
        process_id (int): The ID of the process.
        processed_count (Value): The shared value to store the number of processed items.
        save_dir (str): The directory to save the CSV file.
        save_batch_size (int): The batch size to save to the CSV file.
        sample_rate (int): The sample rate for the audio.
        max_retries (int): The maximum number of retries for processing an item.
    """
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
    """Save a batch of data to the CSV file.

    Args:
        batch (List[dict]): The batch of data to save.
        csv_writer (CSVWriter): The CSV writer to use for saving the data
    """
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
    """Create non-overlapping chunks of the dataset for processing.

    Args:
        dataset (Dataset): The dataset to split into chunks.
        num_workers (int): The number of workers to split the dataset into.
    Returns:
        List[Dataset]: The list of non-overlapping chunks of the dataset."""
    indices = list(range(len(dataset)))
    chunk_size = ceil(len(indices) / num_workers)
    return [
        dataset.select(indices[i : i + chunk_size])
        for i in range(0, len(indices), chunk_size)
    ]


def run_pipeline(
    dataset: Dataset,
    config: dict,
    # devices: List = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    # num_procs_per_device: int = 14,
    # save_dir: str = "./new_audio_v4_2",
    # save_batch_size: int = 10,
    # sample_rate: int = 24_000,
    # max_retries: int = 3,
):
    """Run the pipeline to convert text to audio and tokenize the audio.

    Args:
        dataset (Dataset): The dataset to process.
        devices (List): The list of devices to use for processing.
        num_procs_per_device (int): The number of processes to run on each device.
        save_dir (str): The directory to save the CSV files.
        save_batch_size (int): The batch size to save to the CSV files.
        sample_rate (int): The sample rate for the audio.
        max_retries (int): The maximum number of retries for processing an item."""
    print(config)
    # Unpack the configuration
    (
        devices,
        num_procs_per_device,
        save_dir,
        save_batch_size,
        sample_rate,
        max_retries,
    ) = (
        config["processing"][key]
        for key in [
            "devices",
            "num_procs_per_device",
            "save_dir",
            "save_batch_size",
            "sample_rate",
            "max_retries",
        ]
    )

    # Create the save directory if it does not exist
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
            args=(
                chunk,
                device,
                i,
                processed_count,
                save_dir,
                save_batch_size,
                sample_rate,
                max_retries,
            ),
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
    config = load_config("./synthetic_generation_cfg.yaml")
    logger = configure_logging(config)

    if config["dataset"]["remaining_indices_file"]:
        with open(config["dataset"]["remaining_indices_file"]) as f:
            remaining_indices = json.load(f)
    else:
        logger.info("Process all examples")
        remaining_indices = []

    dataset = load_dataset(
        config["dataset"]["name"], split=config["dataset"]["split"], num_proc=64
    )

    if config["processing"]["do_test"]:
        # Overwrite processing config
        config["processing"]["devices"] = ["cuda:0"]
        config["processing"]["num_procs_per_device"] = 2
        config["processing"]["save_dir"] = "./test"
        config["processing"]["save_batch_size"] = 5
        remaining_indices = range(50)

    if len(remaining_indices) > 0:
        dataset = dataset.select(remaining_indices)

    run_pipeline(dataset, config)
