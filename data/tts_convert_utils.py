"""Utilities for running TTS and conversion to tokens."""
import subprocess
import math
import os


def run_tts(start_idx, end_idx, gpu_id):
    """Run TTS for a given range of indices."""
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python tts.py --start {start_idx} --end {end_idx}"
    subprocess.run(cmd, shell=True, check=True)


def run_convert(start_idx, end_idx, gpu_id, part_id, batch_idx):
    """Run conversion to tokens for a given range of indices.
    
    Args:
        start_idx (int): Start index for the subset
        end_idx (int): End index for the subset
        gpu_id (int): GPU ID
        part_id (int): Part ID for saving the subset
        batch_idx (int): Batch index
    """
    output_dir = f"voice_parts/batch_{batch_idx}"
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python convert_s2tokens.py --start {start_idx} --end {end_idx} --part_id {part_id} --output_dir {output_dir}"
    subprocess.run(cmd, shell=True, check=True)


def process_batch(start_idx, end_idx, gpu_id, part_id, batch_idx, processes_per_gpu):
    """Process a batch of samples.
    
    Args:
        start_idx (int): Start index for the subset
        end_idx (int): End index for the subset
        gpu_id (int): GPU ID
        part_id (int): Part ID for saving the subset
        batch_idx (int): Batch index
        processes_per_gpu (int): Number of processes per GPU
    """
    samples_per_process = math.ceil((end_idx - start_idx + 1) / processes_per_gpu)

    tts_processes = []
    for i in range(processes_per_gpu):
        process_start = start_idx + i * samples_per_process
        process_end = min(process_start + samples_per_process - 1, end_idx)
        p = subprocess.Popen(
            f"CUDA_VISIBLE_DEVICES={gpu_id} python tts.py --start {process_start} --end {process_end}", shell=True
        )
        tts_processes.append(p)

    for p in tts_processes:
        p.wait()

    run_convert(start_idx, end_idx, gpu_id, part_id, batch_idx)
