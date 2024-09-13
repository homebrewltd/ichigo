#!/bin/bash
# Path to your Python script
PYTHON_SCRIPT="synthetic_data_pipeline.py"

# Path to your config file
CONFIG_FILE="configs/synthetic_generation_cfg.yaml"

# Loop from 15 to 24

# echo "Processing batch 0"

# name="/home/root/BachVD/Audio_data/locals/instruction-data-text-only/batch_0/"

# # Construct the save_dir path
# SAVE_DIR="/home/root/BachVD/Audio_data/locals/instruction-speech-whispervq-v3-subset-2/output_0/"


# # Run the Python script with the constructed paths
# python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$name" --save_dir="$SAVE_DIR" --speaker="default_speaker"

# echo "Finished processing batch 0"
# echo "------------------------"

echo "Processing batch 1"

name="/home/root/BachVD/Audio_data/locals/instruction-data-text-only/batch_1/"

# Construct the save_dir path
SAVE_DIR="/home/root/BachVD/Audio_data/locals/instruction-speech-whispervq-v3-subset-2/output_1/"


# Run the Python script with the constructed paths
python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$name" --save_dir="$SAVE_DIR" --speaker="default_speaker"

echo "Finished processing batch 1"
echo "------------------------"

echo "Processing batch 2"

name="/home/root/BachVD/Audio_data/locals/instruction-data-text-only/batch_2/"

# Construct the save_dir path
SAVE_DIR="/home/root/BachVD/Audio_data/locals/instruction-speech-whispervq-v3-subset-2/output_2/"


# Run the Python script with the constructed paths
python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$name" --save_dir="$SAVE_DIR" --speaker="speaker_trump"

echo "Finished processing batch 2"
echo "------------------------"

echo "Processing batch 3"

name="/home/root/BachVD/Audio_data/locals/instruction-data-text-only/batch_3/"

# Construct the save_dir path
SAVE_DIR="/home/root/BachVD/Audio_data/locals/instruction-speech-whispervq-v3-subset-2/output_3/"


# Run the Python script with the constructed paths
python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$name" --save_dir="$SAVE_DIR" --speaker="speaker_trump"

echo "Finished processing batch 3"
echo "------------------------"



echo "All batches processed"