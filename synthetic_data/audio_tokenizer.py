import os
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from encodec import EncodecModel
from encodec.utils import convert_audio
from whisperspeech.vq_stoks import RQBottleneckTransformer
from vq_config import VQConfig
from models.factory import make_vq_model
import os
from datasets import load_dataset

def load_model(
    ref,
    size: str = "medium-vi-2d-2048c-dim64",
    repo_id=None,
    filename=None,
    local_dir=None,
    local_filename=None,
):
    """Load model from file or Hugging Face Hub.

    Args:
        ref (str): Either a local path or "repo_id:filename" format
        repo_id (str, optional): Hugging Face repository ID
        filename (str, optional): Filename in the repository
        local_dir (str, optional): Local directory for downloads
        local_filename (str, optional): Direct path to local file

    Returns:
        RQBottleneckTransformer: Loaded model instance
    """
    # Parse reference string
    if repo_id is None and filename is None and local_filename is None:
        if ":" in ref:
            repo_id, filename = ref.split(":", 1)
        else:
            local_filename = ref

    # Download or use local file
    if not os.path.exists(f"{local_filename}"):
        local_filename = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=local_dir
        )

    # Load and validate spec
    spec = torch.load(local_filename, map_location="cpu")
    model_state_dict = {
        k.replace("model.", ""): v for k, v in spec["state_dict"].items()
    }
    vq_config = VQConfig()
    ichigo_model = make_vq_model(size=size, config=vq_config)
    ichigo_model.load_state_dict(model_state_dict)
    ichigo_model.eval()
    return ichigo_model
class IchigoQuantizer:
    def __init__(self, model_name="jan-hq/ichigo-quantizer:epoch-epoch=8-step=26253-val_epoch_accuracy=0.95.ckpt",model_size="medium-vi-2d-2048c-dim64", device: str = "cuda"):
        """Initialize the Audio Tokenizer with a specified device."""
        self.device = device
        self.ichigo_model = load_model(ref=model_name, size=model_size)
        self.ichigo_model.ensure_whisper(self.device, language='vi')
        self.ichigo_model.to(self.device)
    
    def encode(self, audio: Tuple[torch.Tensor, int]) -> List:
        """Process a single audio file into interleaved audio codes.

        Args:
            audio (Tuple[torch.Tensor, int]): The audio waveform and sample rate.

        Returns:
            List: The interleaved audio codes.
        """
        # Load and pre-process the audio waveform
        wav, sr = audio
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Extract discrete codes
        codes = self.ichigo_model.quantize(wav.to(self.device))
        codes = codes.cpu().tolist()
        return codes

class WhisperVQTokenizer:
    def __init__(self, repo_id: str = "jan-hq/WhisperVQ", device: str = "cuda"):
        """Initialize the Audio Tokenizer with a specified device."""
        # Check if the model is downloaded
        if not os.path.exists("whisper-vq-stoks-v3-7lang-fixed.model"):
            hf_hub_download(
                repo_id=repo_id,
                filename="whisper-vq-stoks-v3-7lang-fixed.model",
                local_dir=".",
            )
        self.device = device
        self.vq_model = RQBottleneckTransformer.load_model(
            "whisper-vq-stoks-v3-7lang-fixed.model"
        ).to(self.device)
        self.vq_model.ensure_whisper(self.device)

    def encode(self, audio: Tuple[torch.Tensor, int]) -> List:
        """Process a single audio file into interleaved audio codes.

        Args:
            audio (Tuple[torch.Tensor, int]): The audio waveform and sample rate.

        Returns:
            List: The interleaved audio codes.
        """
        # Load and pre-process the audio waveform
        wav, sr = audio
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Extract discrete codes
        codes = self.vq_model.encode_audio(wav.to(self.device))
        codes = codes[0].cpu().tolist()
        return codes


class EncodecTokenizer:
    def __init__(self, device: str):
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
    def encode(self, audio: Tuple[torch.Tensor, int]) -> List:
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

if __name__ == "__main__":
    # Load the audio tokenizer
    ds = load_dataset("/home/jan/BachVD/ichigo/synthetic_data/locals/test", split="train")
    # print(len(ds))
    stoks = ds[5]['tokens']
    print(len(stoks))
    print(ds[5]['text'])
    # convert from list to torchtensor
    stoks = torch.tensor(stoks)
    # Load the audio tokenizer
    audio_tokenizer = IchigoQuantizer()
    ichigo_model = audio_tokenizer.ichigo_model
    dequantize_embed = ichigo_model.dequantize(stoks).to(ichigo_model.whmodel[0].device)
    text = ichigo_model.whmodel[0].decode(dequantize_embed, ichigo_model.decoding_options)
    print(text)