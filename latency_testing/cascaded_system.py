import logging
import numpy as np
import torch
import os
import time
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
wav_files= os.listdir("./testing_audio/")
audio_paths = [f"./testing_audio/{file}" for file in wav_files] 
whisper_model_path = "openai/whisper-large-v3"
llama_model_path = "meta-llama/Llama-3.1-8B-Instruct"
whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True, device_map="cuda")
whisper_processor = AutoProcessor.from_pretrained(whisper_model_path)
whisper_pipe      = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                tokenizer=whisper_processor.tokenizer,
                feature_extractor=whisper_processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=1,
                return_timestamps=True,
                torch_dtype=torch.float16,
                device_map="cuda",
            )
whisper_model.eval()

llm_tokenizer           = AutoTokenizer.from_pretrained(llama_model_path, padding_side='left')
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_model               = AutoModelForCausalLM.from_pretrained(llama_model_path, device_map="cuda", torch_dtype=torch.bfloat16)
llm_model.eval()
latencies = []
for audio_path in audio_paths:
    start_time = time.perf_counter()
    whisper_output = whisper_pipe(audio_path, generate_kwargs={"language": "en"})['text'].strip()
    # print(whisper_output)
    querry = """{whisper_output}""" 
    messages = [
        {"role": "user", "content": querry},
    ]
    sample_templated = llm_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
    encoded_batch        = llm_tokenizer(sample_templated, return_tensors="pt").to("cuda")
    generated_ids        = llm_model.generate(**encoded_batch, max_new_tokens=1, pad_token_id=llm_tokenizer.eos_token_id)
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000)
print(f"Latencies: {latencies}")    
latencies.pop(0)
avg_latency = sum(latencies) / len(latencies)
print(f"Mean latency: {np.mean(latencies)} ms")