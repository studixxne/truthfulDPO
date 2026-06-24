import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

from src.utils import get_device, Logger, get_args

@dataclass
class EvalConfig:
    device: str = get_device()
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    peft_model_path: str = None
    use_wandb: bool = False
    project_name: str = "Qwen-alignment"
    run_name: str = None
    run_id: str = None
    batch_size: int = 64

class EvalDataset(Dataset):
    def __init__(self, flatten_data: list[dict]):
        self.flatten_data = flatten_data

    def __len__(self):
        return len(self.flatten_data)
    
    def __getitem__(self, index: int):
        item = self.flatten_data[index]
        return item['question_idx'], item['input_ids'], item['label'], item['mask']

class EvalDynamicCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[tuple]):
        question_idx, input_ids, label, mask = zip(*batch)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)

        return question_idx, padded_input_ids, label, padded_mask

def load_model(base_model_name: str, peft_model_path: str, device: str) -> tuple[torch.Tensor, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float16).to(device)

    if not peft_model_path is None:
        model = PeftModel.from_pretrained(model, peft_model_path).to(device)

    return model, tokenizer

def get_flatten_dataset(tokenizer: AutoTokenizer) -> list[dict]:
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    flatten_data = []

    for i, item in enumerate(dataset):
        question = item['question']
        choices = item['mc2_targets']['choices']
        labels = item['mc2_targets']['labels']

        messages = [{'role': 'user', 'content': question}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        for choice, label in zip(choices, labels):
            choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
            input_ids = torch.tensor(prompt_ids+choice_ids, dtype=torch.long)

            mask = torch.zeros_like(input_ids)
            mask[prompt_len:] = 1
            flatten_data.append({
                'question_idx': i,
                'input_ids': input_ids,
                'label': label,
                'mask': mask
            })

    return flatten_data

def calculate_mc2_score(log_probs: np.array, labels: list[int]) -> float:
    log_probs = log_probs - np.max(log_probs)
    probs = np.exp(log_probs)

    true_probs_sum = sum(p for p, l in zip(probs, labels) if l == 1)
    total_probs_sum = sum(probs)

    mc2_score = true_probs_sum / total_probs_sum
    return mc2_score

def evaluate_truthfulqa_mc(config: EvalConfig):
    from collections import defaultdict

    model, tokenizer = load_model(config.model_name, config.peft_model_path, config.device)
    logger = Logger(config)

    dataset = EvalDataset(get_flatten_dataset(tokenizer))
    collator = EvalDynamicCollator(pad_token_id=tokenizer.pad_token_id)
    eval_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    results = defaultdict(lambda: {"log_probs": [], "labels": []})

    model.eval()
    for question_idx, input_ids, label, mask in tqdm(eval_loader, desc="Evaluating..."):
        input_ids = input_ids.to(config.device)
        mask = mask.to(config.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(config.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_mask = mask[:, 1:]
        shift_input_ids = input_ids[:, 1:]

        log_prob = F.log_softmax(shift_logits, dim=-1)
        token_log_prob = torch.gather(log_prob, dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)

        # (batch, length)
        token_log_prob = token_log_prob * shift_mask
        sum_log_prob = torch.sum(token_log_prob, dim=1)
        lengths = torch.sum(shift_mask, dim=1)
        avg_log_probs = sum_log_prob / lengths

        for i in range(len(question_idx)):
            id = question_idx[i]
            results[id]["log_probs"].append(avg_log_probs[i].item())
            results[id]["labels"].append(label[i])
    
    total_mc2_score = 0.0
    total_question = len(results)

    for item in results.values():
        total_mc2_score += calculate_mc2_score(np.array(item["log_probs"]), item["labels"])
    
    final_mc2_acc = (total_mc2_score / total_question) * 100
    logger.log_mc2_summary(total_question, final_mc2_acc)

if __name__ == "__main__":
    args = get_args(EvalConfig)
    config = EvalConfig(**vars(args))
    evaluate_truthfulqa_mc(config)