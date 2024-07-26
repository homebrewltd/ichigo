import gradio as gr
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import logging
import os

def audio_to_sound_tokens(audio_path, target_bandwidth=1.5, device="cuda"):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(target_bandwidth)
    model.to(device)
    
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    
    audio_code1, audio_code2 = codes[0][0], codes[0][1]
    flatten_tokens = torch.stack((audio_code1, audio_code2), dim=1).flatten().tolist()
    result = ''.join(f'<|sound_{num:04d}|>' for num in flatten_tokens)
    return f'<|sound_start|>{result}<|sound_end|>'

def setup_pipeline(model_path, use_4bit=False, use_8bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_kwargs = {"device_map": "auto"}
    if use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # Adjust this based on your model's tokenizer
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

llm_path = "homebrewltd/llama3-s-2024-07-19"
pipe = setup_pipeline(llm_path, use_8bit=True)
tokenizer = pipe.tokenizer
model = pipe.model

def process_audio(audio_file):
    if audio_file is None:
            raise ValueError("No audio file provided")

    logging.info(f"Audio file received: {audio_file}")
    logging.info(f"Audio file type: {type(audio_file)}")

    sound_tokens = audio_to_sound_tokens(audio_file)
    logging.info("Sound tokens generated successfully")
    # logging.info(f"audio_file: {audio_file.name}")
    messages = [
        {"role": "user", "content": sound_tokens},
    ]

    stop = StopOnTokens()
    input_ids = tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False), return_tensors="pt")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        stopping_criteria=StoppingCriteriaList([stop])
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if tokenizer.eos_token in partial_message:
            break
        yield partial_message
# def stop_generation():
#     # This is a placeholder. Implement actual stopping logic here if needed.
#     return "Generation stopped.", gr.Button.update(interactive=False)
iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload audio")
    ],
    outputs="text",
    title="Llama3-S: A Speech Multimodal Model from Homebrew",
    description="Record or upload a .wav file to generate text based on its content.",
    examples=[
        ["./examples/codeapythonscript.wav"],
        ["./examples/story.wav"],
    ]
)
# iface.load(stop_generation, None, gr.Button("Stop Generation"), queue=False)
iface.launch(share=True)