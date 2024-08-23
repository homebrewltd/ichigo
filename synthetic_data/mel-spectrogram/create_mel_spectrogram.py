import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
import torch.nn.functional as F

def exact_div(x, y):
    assert x % y == 0
    return x // y

# Audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> torch.Tensor:
    """
    Generate the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    
    mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
    mel_filters = torch.from_numpy(mel_filters).to(device)
    
    return mel_filters

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram of an audio file or array.
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    audio = audio.to(device)
    
    # Add pad or trim
    audio = pad_or_trim(audio)
    
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    
    window = torch.hann_window(N_FFT).to(device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def plot_spectrogram(log_mel_spec: torch.Tensor, title: str = "Log Mel Spectrogram"):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(log_mel_spec.cpu().numpy(), aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Frequency Bins")
    plt.tight_layout()
    plt.show()
    

audio_path = "OSR_us_000_0010_8k.wav"
spec = log_mel_spectrogram(audio_path)
plot_spectrogram(spec)
