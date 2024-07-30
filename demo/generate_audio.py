import torchaudio

from whisperspeech.pipeline import Pipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert text to audio.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The text to convert to audio.",
    )
    return parser.parse_args()

def convert_text_to_audio(pipe: Pipeline, text: str):
    """Convert text to audio.

    Args:
        pipe (Pipeline): The pipeline to use for text-to-speech.
        text (str): The text to convert to audio.

    Returns:
        torch.Tensor: The generated audio.
    """
    return pipe.generate(text)


def convert_text_to_audio_file(pipe: Pipeline, text: str, output_path: str):
    """Convert text to audio and save it to a file.

    Args:
        pipe (Pipeline): The pipeline to use for text-to-speech.
        text (str): The text to convert to audio.
        output_path (str): The path to save the audio file.
    """
    pipe.generate_to_file(output_path, text)


class TTSProcessor:
    def __init__(self, device: str):
        """Initialize the TTS Processor with a specified device."""
        self.pipe = Pipeline(
            s2a_ref="collabora/whisperspeech:s2a-q4-tiny-en+pl.model", device=device
        )

    def get_reference_voice_embedding(self, path: str):
        """Get the reference voice embedding from the given audio file.

        Args:
            path (str): The path to the audio file.
        Returns:
            torch.Tensor: The reference voice embedding."""
        return self.pipe.extract_spk_emb(path).cpu()

    def convert_text_to_audio(self, text: str, speaker=None):
        """Convert text to audio.

        Args:
            text (str): The text to convert to audio.

        Returns:
            torch.Tensor: The generated audio.
        """
        return self.pipe.generate(text, speaker=speaker)

    def convert_text_to_audio_file(self, text: str, output_path: str, speaker=None):
        """Convert text to audio and save it to a file.

        Args:
            text (str): The text to convert to audio.
            output_path (str): The path to save the audio file.
        """
        self.pipe.generate_to_file(output_path, text, speaker=speaker)
if __name__ == "__main__":
    args = parse_args()
    processor = TTSProcessor("cuda")
    text = args.text
    text = text.lower()
    text_split = "_".join(text.lower().split(" "))  
    # remove the last character if it is a period
    if text_split[-1] == ".":
        text_split = text_split[:-1]
    print(text_split)
    path = f"./examples/{text_split}.wav"
    processor.convert_text_to_audio_file(text, path)
    