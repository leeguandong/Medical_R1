import os
import torch
import wandb
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    wb_token = ""
    try:
        wandb.login(key=wb_token)
        run = wandb.init(
            project="Lora-R1-Distill-Qwen-14B on Medical COT Dataset",
            job_type="training",
            anonymous="allow",
        )
        return run
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}")
        raise

def load_model_and_tokenizer(config):
    """Load pretrained model and tokenizer with specified configuration."""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_name"],
            local_files_only=True,
            max_seq_length=config["max_seq_length"],
            dtype=config["dtype"],
            load_in_4bit=config["load_in_4bit"],
        )
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {e}")
        raise

def format_prompts(examples, tokenizer, train_prompt_style):
    """Format dataset examples into prompts with EOS token."""
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = [
        train_prompt_style.format(input, cot, output) + tokenizer.eos_token
        for input, cot, output in zip(inputs, cots, outputs)
    ]
    return {"text": texts}

def load_and_prepare_dataset(data_path, tokenizer, train_prompt_style):
    """Load and format the dataset for training."""
    try:
        dataset = load_dataset("json", data_files=data_path, trust_remote_code=True)
        if isinstance(dataset, dict):
            dataset = dataset["train"]
        dataset = dataset.map(
            lambda examples: format_prompts(examples, tokenizer, train_prompt_style),
            batched=True,
        )
        logger.info("Dataset loaded and formatted successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load and prepare dataset: {e}")
        raise

def configure_lora(model, config):
    """Configure LoRA parameters for the model."""
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_r"],
            target_modules=config["lora_target_modules"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config["random_state"],
            use_rslora=False,
            loftq_config=None,
        )
        logger.info("LoRA configuration applied successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to configure LoRA: {e}")
        raise

def main():
    """Main function to orchestrate model training."""
    # Configuration
    config = {
        "model_name": "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": True,
        "data_path": "/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "random_state": 3407,
        "output_dir": "outputs",
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "learning_rate": 2e-4,
        "num_train_epochs": 5,  # Replaced max_steps with epochs
        "logging_steps": 10,
        "weight_decay": 0.01,
        "seed": 8137,
    }

    # Prompt templates
    prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
  请写出恰当完成该请求的回答。
  在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

  ### Instruction:
  你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
  请回答以下医学问题。

  ### Question:
  {}

  ### Response:
  <think>{}"""
    train_prompt_style = prompt_style + """
  </think>
  {}"""

    # Initialize Weights & Biases
    initialize_wandb()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(config["data_path"], tokenizer, train_prompt_style)

    # Configure LoRA
    model = configure_lora(model, config)

    # Initialize trainer
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config["max_seq_length"],
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=config["batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                warmup_steps=config["warmup_steps"],
                learning_rate=config["learning_rate"],
                lr_scheduler_type="linear",
                num_train_epochs=config["num_train_epochs"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=config["logging_steps"],
                optim="adamw_8bit",
                weight_decay=config["weight_decay"],
                seed=config["seed"],
                output_dir=config["output_dir"],
                run_name="medical-o1-sft-experiment",
            ),
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise

    # Train the model
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save the model
    try:
        model.save_pretrained(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        logger.info(f"Model and tokenizer saved to {config['output_dir']}.")
    except Exception as e:
        logger.error(f"Failed to save model and tokenizer: {e}")
        raise

if __name__ == "__main__":
    main()
