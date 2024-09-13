from tts_processor import (
    TTSProcessor,
)  
import uuid

device = "cuda"
tts = TTSProcessor(device)
def text_to_audio_file(text):
    # gen a random id for the audio file
    id = str(uuid.uuid4())
    temp_file = f"./test_audio/{id}_temp_audio.wav"
    text = text
    text_split = "_".join(text.lower().split(" "))  
    # remove the last character if it is a period
    if text_split[-1] == ".":
        text_split = text_split[:-1]
    tts.convert_text_to_audio_file(text, temp_file)
    # logging.info(f"Saving audio to {temp_file}")
    # torchaudio.save(temp_file, audio.cpu(), sample_rate=24000)
    print(f"Saved audio to {temp_file}")
    return temp_file
audio_file = text_to_audio_file("This is the first demo of Whisper Speech, a fully open source text-to-speech model trained by Collabora and Lion on the Juwels supercomputer.")
