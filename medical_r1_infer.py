from unsloth import FastLanguageModel
import torch


def load_model_and_tokenizer(model_path="DeepSeek-R1-Medical-COT", max_seq_length=2048, load_in_4bit=True):
    """Load the fine-tuned model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit
    )
    return model, tokenizer


def generate_response(model, tokenizer, question, max_new_tokens=1200):
    """Generate a response for a given medical question."""
    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}"""

    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [prompt_style.format(question, "")],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True
    )

    response = tokenizer.batch_decode(outputs)[0].split("### Response:")[1]
    return response


def main():
    """Main function to test the fine-tuned model."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Define test question
    question = ("A 61-year-old woman with a long history of involuntary urine loss during activities like "
                "coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. "
                "Based on these findings, what would cystometry most likely reveal about her residual volume "
                "and detrusor contractions?")

    # Generate and print response
    response = generate_response(model, tokenizer, question)
    print("Generated Response:")
    print(response)


if __name__ == "__main__":
    main()