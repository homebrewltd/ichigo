"""TODO: Inference script.

Currently unusable
"""

from typing import Any, Dict, List


def convert_s2tokens(sample):
    # Construct the file path from the sample's audio path info
    file_path = os.path.join(sample["audio"]["path"])

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(file_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to("cuda")
    # wav = wav.unsqueeze(0)

    # Extract discrete codes from the EnCodec model
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

    # Extract the first two sequences from the codes tensor
    audio_code1, audio_code2 = codes[0][0], codes[0][1]

    interleaved = torch.stack((audio_code1, audio_code2), dim=1).flatten()

    # Convert interleaved tensor to list for compatibility with datasets
    return {"tokens": interleaved.tolist()}
