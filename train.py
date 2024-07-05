import torch
import os
import multiprocessing
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    AutoConfig,
)
from trl import SFTTrainer
from datasets import load_dataset
from Adam_mini import Adam_mini


def print_once(message):
    """Print a message only once."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(message)


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model_and_tokenizer(config):
    """Load the model and tokenizer."""
    print_once("--- Load Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"],
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        attn_implementation=config["model"]["attn_implementation"],
        use_cache=False,
        token=config["model"]["token"],
    )

    print_once("--- Load Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"],
        use_fast=True,
        padding_side="right",
        token=config["model"]["token"],
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print_once(f"Pad token ID: {tokenizer.pad_token_id}")
    print_once(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    print_once("--- Initialization complete ---")

    return model, tokenizer


def load_dataset_train(config, num_cores):
    """Load the training dataset."""
    dataset_train = load_dataset(
        config["dataset"]["name"],
        num_proc=num_cores,
        split=config["dataset"]["split"],
    )
    return dataset_train


def print_dataset_info(dataset):
    """Print information about the dataset."""
    print_once("___________________________________")
    print_once(dataset)
    print_once("-----------------------------------")
    print_once(dataset[0]["text"][:100])
    print_once("-----------------------------------")
    print_once(dataset[200]["text"][:100])
    print_once("___________________________________")


class CustomSFTTrainer(SFTTrainer):
    """Custom trainer class."""
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create the optimizer and scheduler."""
        config = AutoConfig.from_pretrained(self.model.config.name_or_path)

        self.optimizer = Adam_mini(
            model=self.model,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            beta1=self.config["optimizer"]["beta1"],
            beta2=self.config["optimizer"]["beta2"],
            epsilon=self.config["optimizer"]["epsilon"],
            zero_3=True,
            n_embd=config.hidden_size,
            n_head=config.num_attention_heads,
            n_query_groups=config.num_key_value_heads,  # GQA
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(
                num_training_steps * self.config["scheduler"]["warmup_ratio"]
            ),
            num_training_steps=num_training_steps,
        )


def calculate_training_steps(dataset, config, num_gpus):
    """Calculate the total number of training steps."""
    total_samples = len(dataset)
    effective_batch_size = (
        config["training"]["per_device_train_batch_size"]
        * num_gpus
        * config["training"]["gradient_accumulation_steps"]
    )
    steps_per_epoch = total_samples // effective_batch_size
    total_steps = steps_per_epoch * config["training"]["num_train_epochs"]
    return total_steps


def print_training_info(config, training_steps, save_steps):
    """Print information about the training process."""
    print_once("___________________________________")
    print_once(
        f"{'Per Device Train Batch Size:':30} {config['training']['per_device_train_batch_size']}"
    )
    print_once(
        f"{'Number of Training Epochs:':30} {config['training']['num_train_epochs']}"
    )
    print_once(
        f"{'Gradient Accumulation Steps:':30} {config['training']['gradient_accumulation_steps']}"
    )
    print_once(f"{'Training steps':30} {training_steps}")
    print_once(f"{'Saving steps':30} {save_steps}")
    print_once("___________________________________")


def check_missing_eos_token_id(dataset_train, data_collator, tokenizer):
    """Check if the EOS token ID is missing."""
    input_ids, attention_mask, labels = data_collator([dataset_train[0]]).values()
    print("check_dataset_labels:")  # noqa
    print(tokenizer.decode(input_ids[0]))  # noqa
    for token, label in zip(input_ids[0], labels[0]):
        print(f"{token.item()}, '{tokenizer.decode(token)}', {label.item()}")
        assert token.item() == label.item(), "Token and label mismatch"


def main():
    """Run the training process."""
    config = load_config("training_config.yaml")

    num_cores = multiprocessing.cpu_count()
    print_once(f"Number of CPU cores: {num_cores}")
    print_once("___________________________________")

    model, tokenizer = load_model_and_tokenizer(config)

    dataset_train = load_dataset_train(config, num_cores)
    print_dataset_info(dataset_train)

    gpu_count = torch.cuda.device_count()

    training_steps = calculate_training_steps(dataset_train, config, gpu_count)
    save_steps = int(training_steps // 80)

    print_training_info(config, training_steps, save_steps)

    # Create the custom trainer
    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        dataset_text_field="text",
        max_seq_length=config["training"]["max_seq_length"],
        dataset_num_proc=num_cores,
        packing=config["training"]["packing"],
        args=TrainingArguments(
            per_device_train_batch_size=config["training"][
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=config["training"][
                "gradient_accumulation_steps"
            ],
            num_train_epochs=config["training"]["num_train_epochs"],
            bf16=config["training_args"]["bf16"],
            logging_steps=config["training_args"]["logging_steps"],
            save_strategy=config["training_args"]["save_strategy"],
            save_steps=save_steps,
            save_total_limit=config["training_args"]["save_total_limit"],
            learning_rate=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
            seed=config["training_args"]["seed"],
            output_dir=config["training_args"]["output_dir"],
            report_to=config["training_args"]["report_to"],
            max_grad_norm=config["training_args"]["max_grad_norm"],
        ),
    )

    collator = trainer.data_collator
    check_missing_eos_token_id(dataset_train, collator, tokenizer)

    trainer_stats = trainer.train(resume_from_checkpoint=False)
    return trainer_stats


if __name__ == "__main__":
    main()
