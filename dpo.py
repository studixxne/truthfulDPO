import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from utils import *

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    project_name: str = "Qwen-alignment"
    device: str = get_device()
    beta: float = 0.1
    max_length: int = 256
    epochs: int = 2
    lr: float = 4e-6
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    batch_size: int = 16
    grad_accum: int = 2
    log_interval: int = 10
    eval_interval: int = 100
    save_dir: str = "./checkpoints"
    save_interval: int = 100

class DPODataset(Dataset):
    def __init__(self, samples: list[tuple[str, str, str]], tokenizer: AutoTokenizer, config: TrainConfig):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = config.max_length

    def _encode(self, prompt, response) -> torch.Tensor:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding="max_length").input_ids
        return input_ids.squeeze(0)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, yw, yl = self.samples[index]

        chosen_ids = self._encode(x, yw)
        rejected_ids = self._encode(x, yl)
        return chosen_ids, rejected_ids        
    
    def __len__(self):
        return len(self.samples)
    
def get_dataloaders(tokenizer: AutoTokenizer, config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    datas = [
        ("사과는 왜 빨개?", "저도 정확하게는 잘 모르지만 ~ 입니다.", "사과는 검정색입니다.")
    ] # 나중에 데이터로 대체

    split = int(len(datas) * 0.95)
    train_dataset = DPODataset(datas[:split], tokenizer, config)
    val_dataset = DPODataset(datas[split:], tokenizer, config)

    is_cuda = (config.device == 'cuda')
    num_workers = 4 if is_cuda else 0

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=is_cuda, persistent_workers=is_cuda)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=is_cuda, persistent_workers=is_cuda)
    
    return train_loader, val_loader


def load_models(config: TrainConfig) -> tuple[nn.Module, nn.Module, AutoTokenizer]:
    device = config.device
    model_name = config.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    for param in ref_model.parameters():
        param.requires_grad = False

    return policy_model, ref_model, tokenizer

def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1) # (batch, seq_len, vocab_size)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # 각 토큰 y에 대한 확률 추출: (batch, seq_len)

    return token_log_probs.sum(dim=-1) # log(pi(y|x))이 -> log(pi(y1|x)*pi(y2|y1)...pi(yn|yn-1) = log(pi(y1|x)) + log(pi(y2|y1)) ... ): (batch, )

def dpo_loss(policy_model: nn.Module, 
             ref_model: nn.Module, 
             chosen_ids: torch.Tensor, 
             rejected_ids: torch.Tensor, 
             config: TrainConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = config.beta

    policy_chosen_logits = policy_model(chosen_ids).logits
    policy_rejected_logits = policy_model(rejected_ids).logits

    with torch.no_grad():
        ref_chosen_logits = ref_model(chosen_ids).logits
        ref_rejected_logits = ref_model(rejected_ids).logits

    chosen_reward = beta * (get_log_probs(policy_chosen_logits, chosen_ids) - get_log_probs(ref_chosen_logits, chosen_ids))
    rejected_reward = beta * (get_log_probs(policy_rejected_logits, rejected_ids) - get_log_probs(ref_rejected_logits, rejected_ids))

    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    return loss, chosen_reward.mean(), rejected_reward.mean()

@torch.no_grad()
def evaluate(policy_model: nn.Module, ref_model: nn.Module, val_loader: DataLoader, config: TrainConfig) -> float:
    policy_model.eval()

    total_step = len(val_loader)
    val_iter = iter(val_loader)

    total_loss = 0.0

    for step in tqdm(range(total_step)):
        yw, yl = next(val_iter)
        yw, yl = yw.to(config.device), yl.to(config.device)

        with autocast_ctx(config.device):
            loss, _, _ = dpo_loss(policy_model, ref_model, yw, yl, config)
        total_loss += loss.item()
    
    policy_model.train()

    return total_loss / total_step

def dpo_train(config: TrainConfig):
    policy_model, ref_model, tokenizer = load_models(config)

    train_loader, val_loader = get_dataloaders(tokenizer, config)
    total_steps = int(len(train_loader) * config.epochs)

    optimizer = get_optimizer(policy_model, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*config.warmup_ratio, num_training_steps=total_steps)

    train_iter = iter(train_loader)
    policy_model.train()

    logger = Logger("Qwen-alignment", True, config)
    best_val_loss = float('inf')

    for step in tqdm(range(total_steps)):
        try:
            yw, yl = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            yw, yl = next(train_iter)

        yw, yl = yw.to(config.device), yl.to(config.device)

        with autocast_ctx(config.device):
            loss, yw_reward, yl_reward = dpo_loss(policy_model, ref_model, yw, yl, config)
            loss = loss / config.grad_accum
        
        loss.backward()

        # update
        if (step + 1) % config.grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # loging
        if (step + 1) % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            loss_log = loss.item() * config.grad_accum
            logger.log_train(step, loss_log, yw_reward, yl_reward, current_lr)

        # eval
        if (step + 1) % config.eval_interval == 0:
            val_loss = evaluate(policy_model, ref_model, val_loader, config)
            logger.log_val(step, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                policy_model.save_pretrained(f"{config.save_dir}/best")
                tokenizer.save_pretrained(f"{config.save_dir}/best")

        # checkpoint
        if (step + 1) % config.save_interval == 0:
            policy_model.save_pretrained(f"{config.save_dir}/step_{step+1}")
            tokenizer.save_pretrained(f"{config.save_dir}/step_{step+1}")

    logger.finish()

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    args = get_args()
    config = TrainConfig(**vars(args))