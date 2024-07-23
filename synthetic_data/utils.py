import os
import logging
from math import ceil
from multiprocessing import Pool

import boto3
import yaml


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


def save_failed_indices(batch_of_failed_indices: list, file_path: str):
    """Save the failed indices to a file."""
    with open(file_path, "w+") as f:
        f.write("\n".join(map(str, batch_of_failed_indices)) + "\n")


def load_config(config_path: str) -> dict:
    """Load the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def upload_file(args):
    client, local_path, s3_path, bucket_name = args
    try:
        client.upload_file(local_path, bucket_name, s3_path)
        print(f"Uploaded {local_path} to {s3_path}")
    except Exception as e:
        print(f"Error uploading {local_path}: {str(e)}")


def upload_folder_to_s3(local_folder, bucket_name, s3_folder, num_processes=4):
    client = boto3.client("s3")

    upload_tasks = []
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_folder, relative_path).replace("\\", "/")
            upload_tasks.append((client, local_path, s3_path, bucket_name))

    with Pool(num_processes) as pool:
        pool.map(upload_file, upload_tasks)


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
