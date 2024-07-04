"""Train the model using the Adam_mini optimizer."""
import multiprocessing
import os
from typing import Any, Dict, List

import torch
from datasets import interleave_datasets, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from trl import SFTTrainer

import Adam_mini

# Constants
MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME = "jan-hq/instruction-speech-conversation"
NEW_SOUND_START = "<|sound_start|>"
NEW_SOUND_END = "<|sound_end|>"
NUM_SOUND_TOKENS = 1024


def print_once(message: str) -> None:
    """Print the message only once."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(message)


def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print_once("--- Loading Model and Tokenizer ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def add_new_tokens(tokenizer, model):
    """Add new tokens to the tokenizer and resize the model's embedding layer.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer
        model (PreTrainedModel): Model
    Return:
        int: Total number of new tokens
    """
    print_once("--- Adding New Tokens ---")
    new_tokens = [f"<|sound_{i:04}|>" for i in range(NUM_SOUND_TOKENS)]
    all_new_tokens = [NEW_SOUND_START, NEW_SOUND_END] + new_tokens

    tokenizer.add_tokens(all_new_tokens, special_tokens=True)
    tokenizer.add_tokens(new_tokens, special_tokens=False)

    total_new_tokens = len(all_new_tokens + new_tokens)
    model.resize_token_embeddings(new_num_tokens=len(tokenizer.get_vocab()))

    print_once(f"Total new tokens: {total_new_tokens}")
    print_once(f"Number of tokens in the vocabulary: {len(tokenizer.get_vocab())}")
    print_once(f"Size of the embedding layer: {len(model.model.embed_tokens.weight.data)}")
    print_once(f"Size of the language model head: {len(model.lm_head.weight.data)}")

    return total_new_tokens


def initialize_new_embeddings(model, total_new_tokens):
    """Initialize the new embeddings using the pre-expansion embeddings.

    Args
        model (PreTrainedModel): Model
        total_new_tokens (int): Total number of new tokens
    Return:
        None
    """
    print_once("--- Initializing New Embeddings ---")
    params = model.state_dict()
    embeddings = params["model.embed_tokens.weight"]
    pre_expansion_embeddings = embeddings[:-total_new_tokens, :]

    mu = torch.mean(pre_expansion_embeddings, dim=0).to(torch.float32)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    sigma = sigma.to(torch.float32)

    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)
    new_embeddings = torch.stack(tuple(dist.sample() for _ in range(total_new_tokens)), dim=0)

    embeddings[-total_new_tokens:, :] = new_embeddings
    params["model.embed_tokens.weight"][-total_new_tokens:, :] = new_embeddings
    model.load_state_dict(params)


def formatting_prompts_func(examples: Dict[str, List[Any]], column_name: str, tokenizer) -> Dict[str, List[str]]:
    """Format the prompts for the model.

    Args:
        examples (Dict[str, List[Any]]): Examples
        column_name (str): Column name
        tokenizer (PreTrainedTokenizer): Tokenizer
    Return:
        Dict[str, List[str]]: Formatted prompts
    """
    convos = examples[column_name]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}


def prepare_dataset(tokenizer):
    """Prepare the training dataset.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer
    Return:
        Dataset: Training dataset
    """
    print_once("--- Preparing Dataset ---")
    dataset = load_dataset(DATASET_NAME, num_proc=multiprocessing.cpu_count(), split="train")
    columns_to_remove = [col for col in dataset.column_names if col not in ["sound_convo", "text_convo"]]
    dataset = dataset.remove_columns(columns_to_remove)

    dataset_sound = dataset.map(
        lambda x: formatting_prompts_func(x, "sound_convo", tokenizer),
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        batch_size=10000,
    ).remove_columns(["sound_convo", "text_convo"])

    dataset_text = dataset.map(
        lambda x: formatting_prompts_func(x, "text_convo", tokenizer),
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        batch_size=10000,
    ).remove_columns(["sound_convo", "text_convo"])

    dataset_train = interleave_datasets(
        [dataset_sound, dataset_text],
        probabilities=[0.8, 0.2],
        seed=42,
        stopping_strategy="first_exhausted",
    )

    return dataset_train


class CustomSFTTrainer(SFTTrainer):
    """Custom Trainer for training the model."""

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create the optimizer and learning rate scheduler.

        Args:
            num_training_steps (int): Number of training steps
        """
        config = AutoConfig.from_pretrained(self.model.config.name_or_path)

        self.optimizer = Adam_mini.Adam_mini(
            model=self.model,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-6,
            zero_3=True,
            n_embd=config.hidden_size,
            n_head=config.num_attention_heads,
            n_query_groups=config.num_key_value_heads,
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps,
        )


def calculate_training_steps(dataset, batch_size, num_gpus, num_epochs, gradient_accumulation_steps):
    """Calculate the total number of training steps.

    Args:
        dataset (Dataset): Training dataset
        batch_size (int): Per device training batch size
        num_gpus (int): Number of GPUs
        num_epochs (int): Number of training epochs
        gradient_accumulation_steps (int): Number of gradient accumulation steps
    Return:
        int: Total number of training steps
    """
    total_samples = len(dataset)
    effective_batch_size = batch_size * num_gpus * gradient_accumulation_steps
    steps_per_epoch = total_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    return total_steps


def main():
    """Training the model."""
    num_cores = multiprocessing.cpu_count()
    print_once(f"Number of CPU cores: {num_cores}")

    model, tokenizer = load_model_and_tokenizer()
    total_new_tokens = add_new_tokens(tokenizer, model)
    initialize_new_embeddings(model, total_new_tokens)

    dataset_train = prepare_dataset(tokenizer)

    # Training parameters
    per_device_train_batch_size = 3
    num_train_epochs = 1
    gradient_accumulation_steps = 8

    print_once(f"Per Device Train Batch Size: {per_device_train_batch_size}")
    print_once(f"Number of Training Epochs: {num_train_epochs}")
    print_once(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

    gpu_count = torch.cuda.device_count()
    training_steps = calculate_training_steps(
        dataset_train, per_device_train_batch_size, gpu_count, num_train_epochs, gradient_accumulation_steps
    )
    save_steps = int(training_steps // 80)

    print_once(f"Training steps: {training_steps}")
    print_once(f"Saving steps: {save_steps}")

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

    trainer_stats = trainer.train(resume_from_checkpoint=True)
    print_once(trainer_stats)
    print_once("Training completed.")


if __name__ == "__main__":
    main()
