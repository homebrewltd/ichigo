import argparse
import os

from datasets import load_dataset
from whisperspeech.pipeline import Pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate TTS from prompts")
    parser.add_argument("--start", type=int, required=True, help="Start index")
    parser.add_argument("--end", type=int, required=True, help="End index")
    return parser.parse_args()


def main(start, end):
    """Generate TTS from prompts and save to audio files.

    Args:
        start (int): Start index of the dataset
        end (int): End index of the dataset
    """
    # Load dataset and filter by index range
    dataset = load_dataset("jan-hq/prompt-voice", split="train", num_proc=64)
    filtered_dataset = dataset.filter(lambda x: start <= x["index"] <= end)

    # Initialize the pipeline
    pipe = Pipeline(s2a_ref="collabora/whisperspeech:s2a-q4-tiny-en+pl.model")

    # Processing function, optimized to check for existing files
    def generate_and_save_audio(text, index):
        audio_dir = "audio"
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        file_path = os.path.join(audio_dir, f"audio_{index}.wav")
        # Check if audio file already exists
        if os.path.exists(file_path):
            print(f"Skipping generation for {file_path} as it already exists.")
            return {"audio_path": file_path}  # Return existing path if audio already generated
        try:
            pipe.generate_to_file(file_path, text)
            return {"audio_path": file_path}
        except Exception as e:
            print(f"Error generating audio for text: {text}, error: {e}")
            return {"audio_path": None}

    # Mapping with custom index from dataset
    processed_data = filtered_dataset.map(lambda x: generate_and_save_audio(x["prompt"], x["index"]), batched=False)
    # Further processing as needed
    print(processed_data.column_names)


if __name__ == "__main__":
    args = parse_args()
    main(args.start, args.end)
