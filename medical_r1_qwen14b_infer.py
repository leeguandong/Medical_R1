import torch
from unsloth import FastLanguageModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(config):
    """Load trained model and tokenizer with specified configuration."""
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

def generate_response(model, tokenizer, prompt, config):
    """Generate response for a given prompt using the model."""
    try:
        FastLanguageModel.for_inference(model)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=config["max_new_tokens"],
            use_cache=True,
        )
        response = tokenizer.batch_decode(outputs)[0]
        return response.split("### Response:")[1]
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        raise

def main():
    """Main function to perform inference with the trained model."""
    # Configuration
    config = {
        "model_name": "outputs",
        "max_seq_length": 2048,
        "dtype": None,
        "load_in_4bit": True,
        "max_new_tokens": 1200,
    }

    # Prompt template
    prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
  请写出恰当完成该请求的回答。
  在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

  ### Instruction:
  你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
  请回答以下医学问题。

  ### Growth:
  {}

  ### Response:
  <think>{}"""

    # Test question
    question = (
        "一名70岁的男性患者因胸痛伴呕吐16小时就医，心电图显示下壁导联和右胸导联ST段抬高0.1~0.3mV，"
        "经补液后血压降至80/60mmHg，患者出现呼吸困难和不能平卧的症状，体检发现双肺有大量水泡音。"
        "在这种情况下，最恰当的药物处理是什么？"
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Generate response
    prompt = prompt_style.format(question, "")
    response = generate_response(model, tokenizer, prompt, config)
    logger.info("Inference completed successfully.")
    print("### 训练后模型推理结果：")
    print(response)

if __name__ == "__main__":
    main()