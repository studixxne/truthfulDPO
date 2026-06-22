import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

from src.utils import get_device, Logger, get_args

@dataclass
class EvalConfig:
    device = get_device()
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    peft_model_path: str = None
    use_wandb: bool = False
    project_name: str = "Qwen-alignment"

def load_model(base_model_name: str, peft_model_path: str, device: str) -> tuple[torch.Tensor, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float16).to(device)

    if not peft_model_path is None:
        model = PeftModel.from_pretrained(model, peft_model_path).to(device)

    return model, tokenizer

def get_choice_log_prob(model: torch.Tensor, tokenizer: AutoTokenizer, question: str, choice: str, device: str) -> float:
    # Batch Size 1
    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
    choice_ids = tokenizer(choice, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    # (1, seq)
    input_ids = torch.cat([prompt_ids, choice_ids], dim=-1)

    with torch.no_grad():
        logits = model(input_ids).logits

    # (seq-1, vocab)
    x_logits = logits[0, :-1, :]
    # (seq-1, )
    y_labels = input_ids[0, 1:]

    # (seq-1, vocab)
    log_probs = F.log_softmax(x_logits, dim=-1)
    # (seq-1, )
    y_log_probs = torch.gather(log_probs, dim=-1, index=y_labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.shape[1]
    choice_log_probs = y_log_probs[prompt_len-1:]

    # 정답 토큰 확률의 평균을 반환
    return choice_log_probs.mean().item()

def calculate_mc2_score(log_probs: np.array, labels: list[int]) -> float:
    log_probs = log_probs - np.max(log_probs)
    probs = np.exp(log_probs)

    true_probs_sum = sum(p for p, l in zip(probs, labels) if l == 1)
    total_probs_sum = sum(probs)

    mc2_score = true_probs_sum / total_probs_sum
    return mc2_score

def evaluate_truthfulqa_mc(config: EvalConfig):
    model, tokenizer = load_model(config.model_name, config.peft_model_path, config.device)
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    logger = Logger(config)

    total_mc2_score = 0.0
    total_question = len(dataset)

    model.eval()
    for step, item in tqdm(enumerate(dataset)):
        question = item['question']
        choices = item['mc2_targets']['choices']
        labels = item['mc2_targets']['labels']

        log_probs = [get_choice_log_prob(model, tokenizer, question, choice, config.device) for choice in choices]
        mc2_score = calculate_mc2_score(np.array(log_probs), labels)
        total_mc2_score += mc2_score
        logger.log_mc2_step(step, total_question, question, mc2_score)

    final_mc2_acc = (total_mc2_score / total_question) * 100
    logger.log_mc2_summary(total_question, final_mc2_acc)

if __name__ == "__main__":
    args = get_args(EvalConfig)
    config = EvalConfig(**vars(args))
    evaluate_truthfulqa_mc(config)