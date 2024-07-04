#!/bin/bash
# Shell script to run tts.py on 4 GPUs with 14 processes each, distributing 550,433 samples evenly.

# Function to run a block of processes on a given GPU
run_on_gpu() {
    local gpu_id=$1
    local start_sample=$2
    local num_samples=$3
    local processes=$4

    local samples_per_process=$(($num_samples / $processes))
    local remainder=$(($num_samples % $processes))
    
    local start_index=$start_sample
    local end_index=0

    # Run each process in the background
    for i in $(seq 0 $(($processes - 1))); do
        if [ $i -lt $remainder ]; then
            end_index=$(($start_index + $samples_per_process)) # One extra sample for the first 'remainder' processes
        else
            end_index=$(($start_index + $samples_per_process - 1))
        fi

        CUDA_VISIBLE_DEVICES=$gpu_id python tts.py --start $start_index --end $end_index &
        start_index=$(($end_index + 1))
    done
}

# Ensure that the script is provided with the total samples
if [ -z "$1" ]; then
    echo "Usage: ./run_tts.sh <total_samples>"
    exit 1
fi

total_samples=$1
total_processes=32
gpus=4
processes_per_gpu=$(($total_processes / $gpus))
samples_per_gpu=$(($total_samples / $gpus))

# Run blocks on each GPU
run_on_gpu 0 0 $samples_per_gpu $processes_per_gpu      # GPU 0
run_on_gpu 1 $(($samples_per_gpu * 1)) $samples_per_gpu $processes_per_gpu  # GPU 1
run_on_gpu 2 $(($samples_per_gpu * 2)) $samples_per_gpu $processes_per_gpu  # GPU 2
run_on_gpu 3 $(($samples_per_gpu * 3 + ($total_samples % $gpus))) $samples_per_gpu $processes_per_gpu  # GPU 3

# Wait for all processes to finish
wait
