import torch
from dataclasses import dataclass, fields
from contextlib import nullcontext
import argparse

import wandb
from tqdm import tqdm

class Color:
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    BLUE  = "\033[94m"
    YELLOW = "\033[93m"
    PURPLE = "\033[95m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

class Logger():
    def __init__(self, config: dataclass):
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb.init(
                project=config.project_name,
                config=vars(config)
            )
        
    def log_train(self, step: int, loss: float, yw_reward: float, yl_reward: float, lr: float):
        log_str = (
            f"{Color.BOLD}[Step {step:05d}]{Color.RESET} | "
            f"{Color.RED}Loss:{Color.RESET} {loss:.4f} | "
            f"{Color.GREEN}YW_Rwd:{Color.RESET} {yw_reward:6.3f} | "
            f"{Color.BLUE}YL_Rwd:{Color.RESET} {yl_reward:6.3f} | "
            f"{Color.CYAN}LR:{Color.RESET} {lr:.2e}"
        )

        tqdm.write(log_str)

        if self.use_wandb:
            wandb.log({"train/loss": loss, "train/yw_reward": yw_reward,"train/yl_reward": yl_reward, "train/lr": lr}, step=step)
        
    def log_val(self, step: int, val_loss: float):    
        log_str = f"✨ {Color.BOLD}{Color.PURPLE}[Validation @ Step {step:05d}]{Color.RESET} ➔ {Color.PURPLE}Val Loss:{Color.RESET} {val_loss:.4f} 🎯"
        tqdm.write(log_str)

        if self.use_wandb:
            wandb.log({"val/loss": val_loss}, step=step)

    def log_bench_judge(self, idx: int, total: int, question: str, answer: str, status: str, comment: str):
        tqdm.write(f"\n{Color.BOLD}[{idx}|{total}] Evaluating{Color.RESET}")
        tqdm.write(f" {Color.CYAN}• Q:{Color.RESET} {question}")
        tqdm.write(f" {Color.PURPLE}• A:{Color.RESET} {answer}")
        
        if status == "CORRECT":
            status_str = f"{Color.GREEN}{Color.BOLD}🟢 CORRECT{Color.RESET}"
        elif status == "INCORRECT":
            status_str = f"{Color.RED}{Color.BOLD}🔴 INCORRECT{Color.RESET}"
        elif status == "OVER_REFUSAL":
            status_str = f"{Color.YELLOW}{Color.BOLD}🟡 OVER_REFUSAL{Color.RESET}"
        else:
            status_str = f"{Color.RED}{Color.BOLD}❌ INVALID{Color.RESET}"

        tqdm.write(f" ➔ Result: {status_str}")
        tqdm.write(f" ➔ Comment: {comment}")
        tqdm.write("-" * 60)

    def log_bench_summary(self, count: dict, total: int):
        """최종 스코어보드를 출력하고 WandB가 켜져 있다면 정확도 지표를 기록"""
        tqdm.write("\n")
        tqdm.write(f"{Color.BOLD}{Color.BLUE}=" * 60)
        tqdm.write(f"EVALUATION RESULT (Total: {total})")
        tqdm.write(f"=" * 60 + Color.RESET)
        tqdm.write(f"{Color.GREEN}CORRECT{Color.RESET}       : {count['CORRECT']}개 ({count['CORRECT']/total*100:.1f}%)")
        tqdm.write(f"{Color.RED}INCORRECT{Color.RESET}     : {count['INCORRECT']}개 ({count['INCORRECT']/total*100:.1f}%)")
        tqdm.write(f"{Color.YELLOW}OVER_REFUSAL{Color.RESET}  : {count['OVER_REFUSAL']}개 ({count['OVER_REFUSAL']/total*100:.1f}%)")
        tqdm.write(f"{Color.BLUE}-" * 60 + Color.RESET)

        if self.use_wandb:
            # 훈련 중 실시간 검증 그래프를 그리기 위해 WandB에 기록
            metrics = {
                "eval/correct_rate": count['CORRECT'] / total,
                "eval/incorrect_rate": count['INCORRECT'] / total,
                "eval/over_refusal_rate": count['OVER_REFUSAL'] / total,
            }
            
            wandb.log(metrics)
    
    def finish(self):
        if self.use_wandb:
            wandb.finish()

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.999)) -> torch.optim.AdamW:
    device = next(model.parameters()).device
    use_fused = (device.type == 'cuda')

    decay, no_decay = [], []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=use_fused)
    return optimizer

def autocast_ctx(device):
    if device == 'cpu':
        return nullcontext()
    
    return torch.amp.autocast(device, dtype=torch.bfloat16)

def get_args(config: dataclass):
    parser = argparse.ArgumentParser()
    for field in fields(config):
        parser.add_argument(f"--{field.name}", type=field.type, default=field.default)
    return parser.parse_args()