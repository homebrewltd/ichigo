import os
import warnings
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, Process, Value
from typing import List

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
    text: str,
    index: int,
    device: str = "cuda:0",
    sample_rate: int = 24_000,
    max_retries: int = 3,
):
    """Convert text to audio and tokenize the audio.
    
    Args:
        text (str): The text to convert to audio.
        index (int): The index of the example.
        device (str, optional): The device to use for processing. Defaults to "cuda:0".
        sample_rate (int, optional): The sample rate of the audio. Defaults to 24_000.
        max_retries (int, optional): The maximum number of retries. Defaults to 3.
    
    Returns:
        dict: A dictionary containing the index, audio, and tokens or None if all attempts failed.
    """
    logger.info(f"Processing index {index}")
    tts_processor = TTSProcessor(device=device)
    audio_tokenizer = AudioTokenizer(device=device)
    for attempt in range(max_retries):
        try:
            audio = tts_processor.convert_text_to_audio(text).cpu()
            audio_tokens = audio_tokenizer.process_single_audio((audio, sample_rate))
            output = {"index": index, "audio": audio, "tokens": audio_tokens}
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for index {index}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All attempts failed for index {index}")
                output = None
    
    del tts_processor
    del audio_tokenizer
    torch.cuda.empty_cache()
    return output


def save_audio_tokens(
    queue: Queue, audio_dir: str, token_dir: str, saved_count
):
    """Save audio and tokens to disk.
    
    Args:
        queue (Queue): The queue containing the audio tokens.
        audio_dir (str): The directory to save the audio.
        token_dir (str): The directory to save the tokens.
        total_examples (int): The total number of examples.
    """
    while True:
        audio_token = queue.get()
        if audio_token is None:
            break

        index = audio_token["index"]
        audio = audio_token["audio"]
        tokens = audio_token["tokens"]
        logger.info(f"Saving index {index} to {audio_dir} and {token_dir}")

        audio_path = os.path.join(audio_dir, f"audio_{index}.wav")
        token_path = os.path.join(token_dir, f"{index}.pt")

        torchaudio.save(audio_path, audio, 24_000)
        torch.save(tokens, token_path)
        with saved_count.get_lock():
            saved_count.value += 1


def run_pipeline(
    dataset: Dataset,
    devices: List[str] = ["cuda:0"],
    num_procs_per_device: int = 5,
    audio_dir: str = "./audio",
    token_dir: str = "./tokens",
):
    """Run the pipeline to convert text to audio and tokenize the audio.
    
    Args:
        dataset (Dataset): The dataset to process.
        devices (List[str], optional): The devices to use for processing. Defaults to ["cuda:0"].
        num_procs_per_device (int, optional): The number of processes per device. Defaults to 8.
        audio_dir (str, optional): The directory to save the audio. Defaults to "./audio".
        token_dir (str, optional): The directory to save the tokens. Defaults to "./tokens".
    """
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(token_dir, exist_ok=True)

    num_processes = len(devices) * num_procs_per_device
    chunk_size = ceil(len(dataset) / num_processes)
    chunks = []
    for i in range(0, len(dataset), chunk_size):
        if i + chunk_size <= len(dataset):
            chunks.append(dataset.select(range(i, i + chunk_size)))
        else:
            chunks.append(dataset.select(range(i, len(dataset))))

    queue = Queue()
    saved_count = Value('i', 0)
    save_process = Process(
        target=save_audio_tokens, args=(queue, audio_dir, token_dir, saved_count)
    )
    save_process.start()

    with tqdm(total=len(dataset), position=0) as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []    
            for i, chunk in enumerate(chunks):
                device = devices[i % len(devices)]
                for example in chunk:
                    text = example["prompt"]
                    index = example["index"]
                    future = executor.submit(process_text, text, index, device)
                    futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                pbar.set_postfix({'Saved': saved_count.value}, refresh=True)
                if result is not None:
                    queue.put(result)

    queue.put(None)
    save_process.join()


if __name__ == "__main__":
    dataset = load_dataset("jan-hq/instruction-speech-v1.5", split="train")
    run_pipeline(dataset.select(range(100)))
