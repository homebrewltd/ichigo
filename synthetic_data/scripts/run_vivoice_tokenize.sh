# Path to your Python script
PYTHON_SCRIPT="audio_to_audio_tokens.py"

# Path to your config file
CONFIG_FILE="configs/audio_to_audio_tokens_cfg.yaml"

for id in {0..3}
do
    echo "Processing batch $id"
    NAME="/home/jan/BachVD/audio_data/viVoice/batch_${id}/"
    # Construct the save_dir path
    SAVE_DIR="/home/jan/BachVD/audio_data/viVoice/output_${id}/"

    
    # Run the Python script with the constructed paths
    python "$PYTHON_SCRIPT" --config_path="$CONFIG_FILE" --name="$NAME"  --save_dir="$SAVE_DIR" 
    
    echo "Finished processing batch $id"
    echo "------------------------"
done

echo "All batches processed"