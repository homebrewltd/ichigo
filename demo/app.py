import gradio as gr
import torch
import torchaudio
from encodec import EncodecModel
from whisperspeech.vq_stoks import RQBottleneckTransformer
from encodec.utils import convert_audio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import logging
import os
from .generate_audio import (
    TTSProcessor,
)  
import uuid
import argparse
from huggingface_hub import hf_hub_download
def parse_arguments():
    parser = argparse.ArgumentParser(description="Host the Gradio WebUI with custom settings")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to run the server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length for generation")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    return parser.parse_args()
args = parse_arguments()


device = "cuda"
if not os.path.exists("whisper-vq-stoks-v3-7lang-fixed.model"):
    hf_hub_download(
        repo_id="jan-hq/WhisperVQ",
        filename="whisper-vq-stoks-v3-7lang-fixed.model",
        local_dir=".",
    )
vq_model = RQBottleneckTransformer.load_model(
        "whisper-vq-stoks-v3-7lang-fixed.model"
    ).to(device)
vq_model.ensure_whisper(device)
tts = TTSProcessor(device)
use_8bit = False    
llm_path = "homebrewltd/Ichigo-llama3.1-s-instruct-v0.3-phase-3"
tokenizer = AutoTokenizer.from_pretrained(llm_path)
model_kwargs = {}
if args.use_8bit:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
    )
elif args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
else:
    model_kwargs["torch_dtype"] = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(llm_path, **model_kwargs).to(device)

def audio_to_sound_tokens_whisperspeech(audio_path):
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vq_model.encode_audio(wav.to('cuda'))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
def audio_to_sound_tokens_whisperspeech_transcribe(audio_path):
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vq_model.encode_audio(wav.to('cuda'))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|reserved_special_token_69|><|sound_start|>{result}<|sound_end|>'
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

def text_to_audio_file(text):
    # gen a random id for the audio file
    id = str(uuid.uuid4())
    # create a user audio folder to store the audio files if it does not exist
    if not os.path.exists("./user_audio"):
        os.makedirs("./user_audio")
    temp_file = f"./user_audio/{id}_temp_audio.wav"
    text = text
    text_split = "_".join(text.lower().split(" "))  
    # remove the last character if it is a period
    if text_split[-1] == ".":
        text_split = text_split[:-1]
    tts.convert_text_to_audio_file(text, temp_file)
    print(f"Saved audio to {temp_file}")
    return temp_file
def process_input(audio_file=None):
    
    for partial_message in process_audio(audio_file):
        yield partial_message
    
def process_transcribe_input(audio_file=None):
    
    for partial_message in process_audio(audio_file, transcript=True):
        yield partial_message
    
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        stop_ids = [tokenizer.eos_token_id, 128009]  # 128009 is the end of turn token
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
def process_audio(audio_file, transcript=False):
    if audio_file is None:
            raise ValueError("No audio file provided")

    logging.info(f"Audio file received: {audio_file}")
    logging.info(f"Audio file type: {type(audio_file)}")

    sound_tokens = audio_to_sound_tokens_whisperspeech_transcribe(audio_file)  if transcript else audio_to_sound_tokens_whisperspeech(audio_file)
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
        max_new_tokens=args.max_seq_len,
        do_sample=False,
        stopping_criteria=StoppingCriteriaList([stop])
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if tokenizer.eos_token in partial_message:
            break
        partial_message = partial_message.replace("assistant\n\n", "")
        yield partial_message

with gr.Blocks() as iface:
    gr.Markdown("# Ichigo-llama3-s: Llama3.1 with listening capabilities")
    gr.Markdown("Record your voice or upload audio and send it to the model.")
    gr.Markdown("Powered by [Homebrew Ltd](https://homebrew.ltd/) | [Read our blog post](https://homebrew.ltd/blog/llama3-just-got-ears)")

    with gr.Row():
        input_type = gr.Radio(["text", "audio"], label="Input Type", value="audio")
        text_input = gr.Textbox(label="Text Input", visible=False)
        audio_input = gr.Audio(label="Audio", type="filepath", visible=True)
    
    convert_button = gr.Button("Convert to Audio", visible=False)
    submit_button = gr.Button("Send")
    transcrip_button = gr.Button("Make Model Transcribe the audio")
    
    text_output = gr.Textbox(label="Generated Text")
    def reset_textbox():
        return gr.update(value="")
    def update_visibility(input_type):
        return (gr.update(visible=input_type == "text"), 
                gr.update(visible=input_type == "text"))
    def convert_and_display(text):
        audio_file = text_to_audio_file(text)
        return audio_file 

    input_type.change(
        update_visibility,
        inputs=[input_type],
        outputs=[text_input, convert_button]
    )

    convert_button.click(
        convert_and_display,
        inputs=[text_input],
        outputs=[audio_input]
    )
    
    submit_button.click(
        process_input,
        inputs=[audio_input],
        outputs=[text_output]
    )
    transcrip_button.click(
        process_transcribe_input,
        inputs=[audio_input],
        outputs=[text_output]
    )
    
iface.queue(max_size=10)
# launch locally
iface.launch(server_name=args.host, server_port=args.port)
