#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="audio_to_audio_tokens.py"

# Path to your config file
CONFIG_FILE="configs/audio_to_audio_tokens_cfg.yaml"

# Loop from 15 to 24
for id in {0..24}
do
    echo "Processing batch $id"
    
    # Construct the remaining_indices_file path
    REMAINING_INDICES_FILE="/home/phong/Workspace/llama3-s/synthetic_data/locals/splitting_audio_indices/train/batch_${id}.json"
    
    # Construct the save_dir path
    SAVE_DIR="./locals/outputs/outputs_${id}"
    
    # Run the Python script with the constructed paths
    python "$PYTHON_SCRIPT" \
        --config_path="$CONFIG_FILE" \
        --remaining_indices_file="$REMAINING_INDICES_FILE" \
        --save_dir="$SAVE_DIR"
    
    echo "Finished processing batch $id"
    echo "------------------------"
done

echo "All batches processed"