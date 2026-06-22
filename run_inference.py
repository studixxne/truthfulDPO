import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from peft import PeftModel
from utils import *

@dataclass
class InferenceConfig:
    device: str = get_device()
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mode: str = 'sample'
    temperature: float = 0.9
    num_generations: int = 2
    do_sample: bool = True
    input_file: str = "./data/generated/synthetic_prompts.json"
    output_file: str = "./data/generated/qwen_responses.json"
    lora_path: str = None

def main():
    args = get_args(InferenceConfig)
    config = InferenceConfig(**vars(args))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float16).to(config.device)

    if not config.lora_path == None:
        model = PeftModel.from_pretrained(model, config.lora_path)
        print("LoRA 적용 완료")

    dataset = None

    if config.mode == 'eval':
        config.temperature = 0.0
        config.num_generations = 1
        config.do_sample = False
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]

    elif config.mode == 'sample':
        if not os.path.exists(config.input_file):
            raise FileNotFoundError(f"Not exist prompts file")
        
        with open(config.input_file, "r", encoding='utf-8') as f:
            dataset = json.load(f)
    
    else:
        raise ValueError("Mode Error!")
    
    prompts = [item["question"] for item in dataset]
    results = inference(model, tokenizer, config, prompts)

    final_outputs = []

    for item, res in zip(dataset, results):
        question = item['question']

        if config.mode == 'eval':
            final_outputs.append({
                "question": question,
                "model_answer": res[0],
                "correct_answers": item['correct_answers'],
                "incorrect_answers": item['incorrect_answers']
            })

        elif config.mode == 'sample':
            final_outputs.append({
                "question": question,
                "response_a": res[0],
                "response_b": res[1]
            })

    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
    with open(config.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_outputs, f, ensure_ascii=False, indent=2)

def inference(model: torch.Tensor, tokenizer: AutoTokenizer, config: InferenceConfig, prompts: list[str]) -> list[list[str]]:
    logger = Logger(config)

    result = []
    total_prompt = len(prompts)

    logger.log_inference_header(config.mode, total_prompt, config.model_name)

    model.eval()
    for step, prompt in tqdm(enumerate(prompts), desc=f"Qwen Processing [{config.mode.upper()}]"):
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors='pt').to(config.device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, num_return_sequences=config.num_generations, max_new_tokens=256, do_sample=config.do_sample, temperature=config.temperature)

        input_len = model_inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_len:]
        responses = list(map(lambda x: str.strip(x), tokenizer.batch_decode(generated_ids, skip_special_tokens=True)))
        result.append(responses)

        del model_inputs, generated_ids
        if step % 50 == 0:
            if config.device == 'mps':
                torch.mps.empty_cache()
            elif config.device == 'cuda':
                torch.cuda.empty_cache()

        logger.log_inference_step(step+1, total_prompt, prompt, config.mode, responses)

    logger.log_inference_summary(total_prompt, config.output_file)
    
    return result

if __name__ == '__main__':
    main()