from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import torchaudio
from vq_stoks import RQBottleneckTransformer
from vllm import LLM, SamplingParams
import os
from huggingface_hub import hf_hub_download


device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists("whisper-vq-stoks-medium-en+pl-fixed.model"):
    hf_hub_download(
        repo_id="jan-hq/WhisperVQ",
        filename="whisper-vq-stoks-medium-en+pl-fixed.model",
        local_dir=".",
    )
vq_model = RQBottleneckTransformer.load_model(
        "whisper-vq-stoks-medium-en+pl-fixed.model"
    ).to(device)

def audio_to_sound_tokens(audio_path, target_bandwidth=1.5, device=device):
    vq_model.ensure_whisper(device)
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vq_model.encode_audio(wav.to(device))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|reserved_special_token_69|><|sound_start|>{result}<|sound_end|>'
def setup_pipeline(model_path, use_4bit=False, use_8bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = {"device_map": "auto"}

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
llm_path = "homebrewltd/llama3.1-s-instruct-v0.2"
pipe = setup_pipeline(llm_path, use_8bit=False)
sound_tokens = audio_to_sound_tokens("./examples_audio/what-is-the-color-of-the-ocean.wav")
print(sound_tokens)
messages = [
    {"role": "user", "content": sound_tokens},
]
generated_text = generate_text(pipe, messages)

print("-"*50)
print("# Model Output: ", generated_text)