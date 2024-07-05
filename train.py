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

from Adam_mini import Adam_mini


def print_once(message):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(message)


num_cores = multiprocessing.cpu_count()
print_once(f"Number of CPU cores: {num_cores}")
print_once("___________________________________")

# Model loading
print_once("--- Load Model ---")
model_path = "jan-hq/llama-3-sound-init"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
    token="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",
)


# Tokenizer loading
print_once("--- Load Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    padding_side="right",
    token="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",
)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
print_once(tokenizer.pad_token_id)
print_once(len(tokenizer.get_vocab()))
print_once("--- Initialization complete ---")

# Setting up data train
dataset_train = load_dataset(
    "jan-hq/instruction-speech-conversation-interleaved",
    num_proc=num_cores,
    split="train",
)
# def formatting_prompts_func(examples, column_name):
#     convos = examples[column_name]
#     texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
#     return {"text": texts}

# dataset = load_dataset("jan-hq/instruction-speech-conversation", num_proc=num_cores, split="train")
# columns_to_remove = [col for col in dataset.column_names if col not in ["sound_convo", "text_convo"]]
# dataset = dataset.remove_columns(columns_to_remove)
# print_once(dataset)

# dataset_sound = dataset.map(
#     lambda x: formatting_prompts_func(x, "sound_convo"),
#     batched=True,
#     num_proc=num_cores,
#     batch_size=10000
# ).remove_columns(["sound_convo","text_convo"])
# print_once(dataset_sound)

# dataset_text = dataset.map(
#     lambda x: formatting_prompts_func(x, "text_convo"),
#     batched=True,
#     num_proc=num_cores,
#     batch_size=10000
# ).remove_columns(["sound_convo","text_convo"])
# print_once(dataset_text)

# dataset_train = interleave_datasets([dataset_sound, dataset_text], probabilities=[0.8, 0.2], seed=42, stopping_strategy="first_exhausted")

print_once("___________________________________")
print_once(dataset_train)
print_once("-----------------------------------")
print_once(dataset_train[0]["text"][:100])
print_once("-----------------------------------")
print_once(dataset_train[200]["text"][:100])
print_once("___________________________________")


class CustomSFTTrainer(SFTTrainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        config = AutoConfig.from_pretrained(self.model.config.name_or_path)

        self.optimizer = Adam_mini(
            model=self.model,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-6,
            zero_3=True,
            n_embd=config.hidden_size,
            n_head=config.num_attention_heads,
            n_query_groups=config.num_key_value_heads,  # GQA
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),  # 10% warmup
            num_training_steps=num_training_steps,
        )


# Training args
per_device_train_batch_size = 3
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
print_once(f"{'Training steps':30} {training_steps}")
print_once(f"{'Saving steps':30} {save_steps}")
print_once("___________________________________")

# Create the custom trainer
trainer = CustomSFTTrainer(
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
        learning_rate=5e-5,
        weight_decay=0.01,
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",
        max_grad_norm=1,
    ),
)

trainer_stats = trainer.train(resume_from_checkpoint=False)
