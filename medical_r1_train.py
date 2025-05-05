import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


def setup_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    wb_token = ""
    wandb.login(key=wb_token)
    run = wandb.init(
        project="Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset",
        job_type="training",
        anonymous="allow"
    )
    return run


def load_model_and_tokenizer(max_seq_length=2048, load_in_4bit=True):
    """Load the pre-trained model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/dev_share/gdli7/models/LLM/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit
    )
    return model, tokenizer


def format_prompts(examples, train_prompt_style, eos_token):
    """Format dataset prompts for training."""
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = [
        train_prompt_style.format(input, cot, output) + eos_token
        for input, cot, output in zip(inputs, cots, outputs)
    ]
    return {"text": texts}


def prepare_dataset(tokenizer):
    """Load and preprocess the dataset."""
    train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

    dataset = load_dataset(
        "/dev_share/gdli7/datasets/datasets--FreedomIntelligence--medical-o1-reasoning-SFT",
        "en",
        split="train",
        trust_remote_code=False
    )
    dataset = dataset.map(
        lambda examples: format_prompts(examples, train_prompt_style, tokenizer.eos_token),
        batched=True
    )
    return dataset


def configure_peft_model(model):
    """Configure the model with PEFT settings."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    return model


def setup_trainer(model, tokenizer, dataset, max_seq_length):
    """Set up the SFTTrainer for training."""
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=5,  # Train for 1 epoch; adjust as needed
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=training_args
    )
    return trainer


def save_model(model, tokenizer, save_path="DeepSeek-R1-Medical-COT"):
    """Save the fine-tuned model and tokenizer."""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")


def main():
    """Main function to orchestrate model training."""
    # Initialize Weights & Biases
    setup_wandb()

    # Load model and tokenizer
    max_seq_length = 2048
    model, tokenizer = load_model_and_tokenizer(max_seq_length)

    # Prepare dataset
    dataset = prepare_dataset(tokenizer)

    # Configure PEFT
    model = configure_peft_model(model)

    # Set up trainer
    trainer = setup_trainer(model, tokenizer, dataset, max_seq_length)

    # Train model
    trainer.train()

    # Save model
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
