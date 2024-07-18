import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

from trl import SFTTrainer
import multiprocessing
from datasets import load_dataset
from transformers import AutoConfig

def print_once(message):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(message)

num_cores = multiprocessing.cpu_count()
print_once(f"Number of CPU cores: {num_cores}")
print_once("___________________________________")

# Model loading
print_once("--- Load Model ---")
model_path = "jan-hq/Jan-Llama3-0708"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
)

# Tokenizer loading
print_once("--- Load Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    padding_side="right",
)

# change pad token to reserve space for special tokens
# set 128023 as pad token
tokenizer.pad_token_id = 128023
tokenizer.pad_token = tokenizer.convert_ids_to_tokens(128023)
print_once(tokenizer.pad_token_id)
print_once(len(tokenizer.get_vocab()))
print_once("--- Initialization complete ---")

# Setting up data train
dataset_train = load_dataset(
    "jan-hq/instruction-speech-conversation-v1.5-phase-2-sound-convo",
    num_proc=num_cores,
    split="train",
)

print_once("--- Dataset loading ---")
print_once("___________________________________")
print_once(dataset_train)
print_once("-----------------------------------")
print_once(dataset_train[0]["text"][:100])
print_once("-----------------------------------")
print_once(dataset_train[200]["text"][:100])
print_once("___________________________________")

# Training args
per_device_train_batch_size = 4
num_train_epochs = 1
gradient_accumulation_steps = 4

print_once("___________________________________")
print_once(f"{'Per Device Train Batch Size:':30} {per_device_train_batch_size}")
print_once(f"{'Number of Training Epochs:':30} {num_train_epochs}")
print_once(f"{'Gradient Accumulation Steps:':30} {gradient_accumulation_steps}")

config = AutoConfig.from_pretrained(model_path)
gpu_count = torch.cuda.device_count()

def training_step_calc(
    dataset, batch_size, num_gpus, num_epochs, gradient_accumulation_steps
):
    total_samples = len(dataset)
    effective_batch_size = batch_size * num_gpus * gradient_accumulation_steps
    steps_per_epoch = total_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    return total_steps


training_steps = training_step_calc(
    dataset=dataset_train,
    batch_size=per_device_train_batch_size,
    num_gpus=gpu_count,
    num_epochs=num_train_epochs,
    gradient_accumulation_steps=gradient_accumulation_steps,
)

save_steps = int(training_steps // 80)
warmup_steps = int(training_steps*0.05)

print_once(f"{'Training steps':30} {training_steps}")
print_once(f"{'Saving steps':30} {save_steps}")
print_once(f"{'Warming steps':30} {warmup_steps}")
print_once("___________________________________")

# Create the custom trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    dataset_text_field="text",
    max_seq_length=4096,
    dataset_num_proc=num_cores,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        warmup_steps=warmup_steps,
        learning_rate=5e-5,
        weight_decay=0.01,
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",
        max_grad_norm=1,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        hub_model_id="jan-hq/Jan-Llama3-0719",
        push_to_hub=True,
        dataloader_num_workers=16,
        hub_token = "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",
    ),
)

trainer_stats = trainer.train(resume_from_checkpoint=False)
