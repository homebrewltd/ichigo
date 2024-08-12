import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
from typing import Union

# Audio hyperparameters (matching Whisper)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_MELS = 80
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """Load an audio file and return as float32 numpy array."""
    audio, _ = librosa.load(file, sr=sr)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    return audio

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

def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor]):
    """
    Compute the log-Mel spectrogram of an audio file or array.
    
    Parameters:
    - audio: Path to audio file or numpy array of audio data
    
    Returns:
    - torch.Tensor: Log-Mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    audio = pad_or_trim(audio)
    
    window = torch.hann_window(N_FFT)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS))
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec

def plot_spectrogram(log_mel_spec: torch.Tensor, title: str = "Log Mel Spectrogram"):
    """Plot the log Mel spectrogram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(log_mel_spec.numpy(), aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Frequency Bins")
    plt.tight_layout()
    plt.show()

def create_and_plot_spectrogram(audio_path: str):
    spec = log_mel_spectrogram(audio_path)
    plot_spectrogram(spec)
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Processed audio length: {N_SAMPLES / SAMPLE_RATE:.2f} seconds")

# Example usage
create_and_plot_spectrogram("path/to/file/.wav")
