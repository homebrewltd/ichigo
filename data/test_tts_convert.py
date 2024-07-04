"""Test script for run_tts_convert.py."""
import subprocess
import os


def run_test():
    """Run test for run_tts_convert.py."""
    # Test parameters
    total_samples = 100
    batch_size = 20
    num_gpus = 2
    processes_per_gpu = 2

    # Run the script with test parameters
    cmd = f"python run_tts_convert.py --total_samples {total_samples} --batch_size {batch_size} --num_gpus {num_gpus} --processes_per_gpu {processes_per_gpu}"

    print("Running test with the following parameters:")
    print(f"Total Samples: {total_samples}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Processes per GPU: {processes_per_gpu}")

    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\nScript executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while running the script: {e}")
        return

    # Check if output directories were created
    expected_batches = (total_samples + batch_size - 1) // batch_size
    for i in range(expected_batches):
        batch_dir = f"voice_parts/batch_{i}"
        if os.path.exists(batch_dir):
            print(f"Output directory {batch_dir} was created successfully.")
        else:
            print(f"Error: Output directory {batch_dir} was not created.")

    print("\nTest completed. Please check the output above for any errors or missing directories.")


if __name__ == "__main__":
    run_test()
