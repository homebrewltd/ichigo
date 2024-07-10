import os
from typing import List, Tuple

import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

class AudioTokenizer:
    def __init__(self, device: str = "cuda:0"):
        """Initialize the Audio Tokenizer with a specified device."""
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(1.5)
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def decode(self, audio_code: List) -> Tuple[torch.Tensor, int]:
        """Decode interleaved audio codes into audio.

        Args:
            audio_code (List): The interleaved audio codes.

        Returns:
            Tuple[torch.Tensor, int]: The decoded audio and sample rate.
        """
        try:
            # Convert the audio code to tensor and reshape
            interleaved = torch.tensor(audio_code).view(-1, 2)
            audio_code1, audio_code2 = interleaved[:, 0], interleaved[:, 1]
            # Decode the audio code
            codes = (
                torch.stack((audio_code1, audio_code2), dim=0)
                .unsqueeze(0)
                .to(self.device)
            )
            decoded_audio = self.model.decoder(
                self.model.quantizer.decode(codes.transpose(0, 1))
            )
            decoded_audio = decoded_audio.squeeze(0)
            return decoded_audio, self.model.sample_rate
        except Exception as e:
            print(f"Failed to decode audio code: {e}")
            return None

    @torch.no_grad()
    def process_single_audio(self, audio: Tuple[torch.Tensor, int]) -> List:
        """Process a single audio file into interleaved audio codes.

        Args:
            audio (Tuple[torch.Tensor, int]): The audio waveform and sample rate.

        Returns:
            List: The interleaved audio codes.
        """
        # Load and pre-process the audio waveform
        wav, sr = audio
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0).to(self.device)
        # Extract discrete codes from the EnCodec model
        encoded_frames = self.model.encode(wav)
        # Concatenate the encoded frames
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

        # Extract the first two sequences from the codes tensor
        audio_code1, audio_code2 = codes[0][0], codes[0][1]

        # Interleave the audio codes between low and high frequency bands
        interleaved = torch.stack((audio_code1, audio_code2), dim=1).flatten()
        return interleaved.tolist()


# class AudioTokenizerGroup:
#     def __init__(
#         self, num_processes_per_gpu, gpu_ids, output_dir: str = "./tokenized_sound_data"
#     ):
#         self.num_processes = num_processes_per_gpu
#         self.gpu_ids = gpu_ids
#         self.output_dir = output_dir
#         self.failed_idx = []
#         self.completed_outputs = {}

#         # Prepare folders and distribute pipeline
#         self._prepare_folders()
#         print("Setting up", len(self._distribute_encodecs()))

#     def _prepare_folders(self):
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)

#     def _distribute_encodecs(self):
#         """Distribute the pipeline."""
#         self.encodecs = []

#         for gpu_id in self.gpu_ids:
#             for _ in range(self.num_processes):
#                 self.encodecs.append(AudioTokenizer(device=f"cuda:{gpu_id}"))

#     def tokenize_audio(self, audio_idx: List[int], audio_dir: str = "./audio"):
#         """Tokenize audio files into interleaved audio codes.

#         Args:
#             audio_idx (List[int]): The list of audio indices.
#             audio_dir (str): The directory containing the audio files.
#         """
#         # Get all audio paths using the audio_idx
#         audio_paths = [os.path.join(audio_dir, f"audio_{idx}.wav") for idx in audio_idx]

#         # Batch process based on the number of encodecs using multiprocessing
#         total_num_encodecs = len(self.encodecs)
#         with ThreadPoolExecutor(max_workers=total_num_encodecs) as executor:
#             for index, audio_path in enumerate(audio_paths):
#                 encodec = self.encodecs[index % total_num_encodecs]
#                 executor.submit(
#                     self.single_encode, audio_path, encodec, audio_idx[index]
#                 )

#     def single_encode(self, audio_path, encodec, index):
#         """Process a single audio file.

#         Args:
#             audio_path (str): The path to the audio file.
#             encodec (EncodecModel): The encodec model to use for processing.
#             index (int): The index of the audio file.
#         """
#         try:
#             # Process the audio file if successful
#             sound_tokens = encodec.process_single_audio(audio_path)
#             return {"index": index, "input_token_sound": sound_tokens}
#         except Exception as e:
#             print(f"Error processing audio file: {audio_path}, error: {e}")
#             self.failed_idx.append(index)
