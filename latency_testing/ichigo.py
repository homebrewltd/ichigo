import time
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from whisperspeech.vq_stoks import RQBottleneckTransformer
import os
def setup_pipeline(model_path, use_4bit=False, use_8bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = {"device_map": "cuda"}

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    return pipeline("text-generation", model=model, tokenizer=tokenizer)
device = "cuda"
vqmodel = RQBottleneckTransformer.load_model(
        "whisper-vq-stoks-v3-7lang-fixed.model"
    ).to(device)
vqmodel.ensure_whisper(device)
tokenizer = AutoTokenizer.from_pretrained("homebrewltd/Ichigo-llama3.1-s-instruct-v0.3-phase-3")
llm_path = "homebrewltd/Ichigo-llama3.1-s-instruct-v0.3-phase-3"
pipe = setup_pipeline(llm_path, use_8bit=False)
def audio_to_sound_tokens(audio_path):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vqmodel.encode_audio(wav.to("cuda"))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
wav_files= os.listdir("./testing_audio/")
audio_paths = [f"./testing_audio/{file}" for file in wav_files] 
def generate_text(pipe, messages, max_new_tokens=64, temperature=0.0, do_sample=False):
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "temperature": temperature,
        "do_sample": do_sample,
    }

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Usage
def measure_latency(pipe, audio_paths):
    latencies = []
    latencies_enc = []
    for audio_path in audio_paths:

        start_time_2 = time.perf_counter()
        sound_tokens = audio_to_sound_tokens(audio_path)
        end_time_2 = time.perf_counter()
        latency_enc = (end_time_2 - start_time_2) * 1000
        latencies_enc.append(latency_enc)
        start_time = time.perf_counter()
        messages = [
                {"role": "user", "content": sound_tokens},
            ]
        with torch.no_grad():
            generated_text = generate_text(pipe, messages, max_new_tokens=1)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)
    print(f"Latencies: {latencies}")
    latencies.pop(0)
    latencies_enc.pop(0)
    avg_latency_enc = sum(latencies_enc) / len(latencies_enc)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, avg_latency_enc

avg_latency, avg_latency_enc = measure_latency(pipe, audio_paths)
print(f"Average time to first token: {avg_latency+avg_latency_enc:.2f} ms")
print(f"Average time to encode audio: {avg_latency_enc:.2f} ms")