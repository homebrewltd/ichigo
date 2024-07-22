from typing import List, Tuple

import torch
from encodec import EncodecModel
from encodec.utils import convert_audio


def encode(
    model: EncodecModel, audio: Tuple[torch.Tensor, int], device: str
) -> List[int]:
    """Encode audio into interleaved audio codes.

    Args:
        model (EncodecModel): The EnCodec model.
        audio (Tuple[torch.Tensor, int]): The audio waveform and sample rate.
        device (str): The device to use.
    Return:
        List: The interleaved audio codes.
    """
    # Send model to device
    model.to(device)

    # Load and pre-process the audio waveform
    wav, sr = audio
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)

    # Extract discrete codes from the EnCodec model
    encoded_frames = model.encode(wav)
    # Concatenate the encoded frames
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

    # Extract the first two sequences from the codes tensor
    audio_code1, audio_code2 = codes[0][0], codes[0][1]

    # Interleave the audio codes between low and high frequency bands
    interleaved = torch.stack((audio_code1, audio_code2), dim=1).flatten()
    return interleaved.tolist()


def decode(
    model: EncodecModel, audio_code: List, device: str
) -> Tuple[torch.Tensor, int]:
    """Decode interleaved audio codes into audio.

    Args:
        model (EncodecModel): The EnCodec model.
        audio_code (List): The interleaved audio codes.
        device (str): The device to use.
    Return:
        Tuple[torch.Tensor, int]: The decoded audio and sample rate.
    """
    # Send model to device
    model.to(device)

    # Convert the audio code to tensor and reshape
    interleaved = torch.tensor(audio_code).view(-1, 2)
    audio_code1, audio_code2 = interleaved[:, 0], interleaved[:, 1]
    # Decode the audio code
    codes = torch.stack((audio_code1, audio_code2), dim=0).unsqueeze(0).to(device)
    decoded_audio = model.decoder(model.quantizer.decode(codes.transpose(0, 1)))
    decoded_audio = decoded_audio.squeeze(0)
    return decoded_audio, model.sample_rate


class AudioTokenizer:
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
