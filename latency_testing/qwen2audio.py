from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import os
import time
import torch
# allow only cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
wav_files= os.listdir("./testing_audio/")
audio_paths = [f"./testing_audio/{file}" for file in wav_files] 
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda:0")
latencies = []  
for audio_path in audio_paths:
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": f"{audio_path}"},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    # print(text)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        ele['audio_url'], 
                        sr=processor.feature_extractor.sampling_rate)[0]
                    )
    start_time = time.perf_counter()
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda:0")
    inputs = inputs.to("cuda:0")

    generate_ids = model.generate(**inputs,  max_new_tokens=1)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000)
# print(f"Latencies: {latencies}")
# pop the first element cause model need first run to warm up
latencies.pop(0)
latency = sum(latencies) / len(latencies)   
print(f"Latencies: {latency}")

