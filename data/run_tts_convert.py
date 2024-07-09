"""Run TTS and conversion to tokens in batches."""

import argparse
import math
import subprocess
from tqdm import tqdm

# from tts_convert_utils import process_batch
import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-GPU TTS and Conversion")
    parser.add_argument("--total_samples", type=int, help="Total number of samples to process", default=700000)
    parser.add_argument("--batch_size", type=int, help="Number of samples to process in each batch", default=10000)
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--processes_per_gpu", type=int, default=14, help="Number of processes per GPU")
    parser.add_argument("--restore_from_batch", type=int, default=-1, help="Batch to start from")
    return parser.parse_args()


def main():
    """Run TTS and conversion to tokens in batches."""
    args = parse_arguments()
    print("Arguments: ", args)

    num_batches = math.ceil(args.total_samples / args.batch_size)
    start_batch = args.restore_from_batch

    with tqdm(total=num_batches, desc="Processing batches") as pbar:
        for batch in range(num_batches):
            start_idx = batch * args.batch_size
            end_idx = min((batch + 1) * args.batch_size - 1, args.total_samples - 1)
            if batch >= start_batch: # Skip batches before the restore_from_batch
                processes = []
                samples_per_gpu = math.ceil((end_idx - start_idx + 1) / args.num_gpus)

                for gpu in range(args.num_gpus):
                    gpu_start = start_idx + gpu * samples_per_gpu
                    gpu_end = min(gpu_start + samples_per_gpu - 1, end_idx)

                    p = subprocess.Popen(
                        [
                            "python",
                            "-c",
                            f"from tts_convert_utils import process_batch; process_batch({gpu_start}, {gpu_end}, {gpu}, {batch * args.num_gpus + gpu}, {batch}, {args.processes_per_gpu})",
                        ]
                    )
                    processes.append(p)

                for p in processes:
                    p.wait()

            else: 
                print("Skipping batch", batch)

            pbar.update(1)
            pbar.set_postfix({"Last processed": end_idx})


if __name__ == "__main__":
    main()
