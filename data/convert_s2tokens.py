import os
import re
import torch
import torchaudio
from datasets import load_dataset, DatasetDict, Audio
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()#src/datasets/utils/logging.py
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from encodec import EncodecModel
from encodec.utils import convert_audio
import argparse

import warnings
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser(description='Process audio files and convert to tokens.')
parser.add_argument('--start', type=int, help='Start index for the subset')
parser.add_argument('--end', type=int, help='End index for the subset')
parser.add_argument('--part_id', type=int, help='Part ID for saving the subset')
parser.add_argument('--output_dir', type=str, help='Output directory for processed data')

args = parser.parse_args()


# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define the directory containing the audio files and pattern
directory = 'audio'
pattern = r'audio_(\d+)\.wav'

# Get all files in the directory
files = os.listdir(directory)
indices = [int(match.group(1)) for file in files if (match := re.match(pattern, file))]

# Print all indices
print(len(indices), indices[0])

# Load and filter the dataset
dataset = load_dataset("jan-hq/instruction-speech-v1.5", split='train')
list_index = set(indices)
filtered_dataset = dataset.filter(lambda batch: [idx in list_index for idx in batch['index']], batched=True, num_proc=64) # Filter the dataset

print(filtered_dataset)

# Add audio paths to the dataset
def add_audio_path(example):
    example['audio'] = f"audio/audio_{example['index']}.wav"
    return example
    
dataset = filtered_dataset.map(add_audio_path, num_proc=64)
print(dataset[0])

# Ensure audio column is properly configured
if 'audio' in dataset.column_names:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
print(dataset[0])

# Add tokens
llama_tokenizer = PreTrainedTokenizerFast.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
llama_tokenizer.add_tokens("<|sound_start|>",special_tokens=True)
llama_tokenizer.add_tokens("<|sound_end|>",special_tokens=True)
for sound_token in range(0, 1024):
    sound_added_token = f"<|sound_{sound_token:04}|>"
    llama_tokenizer.add_tokens(sound_added_token)
pad_idx = llama_tokenizer.vocab['<|sound_0000|>']
pos_sound_start = llama_tokenizer.vocab['<|sound_start|>']
pos_sound_end = llama_tokenizer.vocab['<|sound_end|>']

print("--- Tokenizer set up ---")
      
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(1.5)
model.to("cuda")

def process_audio(sample):
    # Construct the file path from the sample's audio path info
    file_path = os.path.join(sample['audio']['path'])

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(file_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to("cuda")

    # Extract discrete codes from the EnCodec model
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

    # Extract the first two sequences from the codes tensor
    audio_code1, audio_code2 = codes[0][0], codes[0][1]

    interleaved = torch.stack((audio_code1, audio_code2), dim=1).flatten()

    # Convert interleaved tensor to list for compatibility with datasets
    return {'tokens': interleaved.tolist()}

# Apply the function using map with batching and multiple processes
subset = dataset.select(range(args.start, args.end))
print(subset)
processed_dataset = subset.map(process_audio, batched=False, num_proc=1)

# Example output
print(processed_dataset)

saved_path = os.path.join(args.output_dir, f"part_{args.part_id}")
# Saving the processed dataset to the specified output directory
processed_dataset.save_to_disk(saved_path)
print(f"Processed dataset saved to {saved_path}")