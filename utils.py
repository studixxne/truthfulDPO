import torch
from dataclasses import dataclass, fields
from contextlib import nullcontext
import argparse

import wandb
import tqdm

class Logger():
    def __init__(self, project_name: str, use_wandb: bool, config):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                config=vars(config)
            )

    def log_train(self, step: int, loss: float, yw_reward: float, yl_reward: float, lr: float):
        CYAN  = "\033[96m"
        GREEN = "\033[92m"
        RED   = "\033[91m"
        BLUE  = "\033[94m"
        RESET = "\033[0m"
        BOLD  = "\033[1m"

        log_str = (
            f"{BOLD}[Step {step:05d}]{RESET} | "
            f"{RED}Loss:{RESET} {loss:.4f} | "
            f"{GREEN}YW_Rwd:{RESET} {yw_reward:6.3f} | "
            f"{BLUE}YL_Rwd:{RESET} {yl_reward:6.3f} | "
            f"{CYAN}LR:{RESET} {lr:.2e}"
        )

        tqdm.write(log_str)

        if self.use_wandb:
            wandb.log({"train/loss": loss, "train/yw_reward": yw_reward,"train/yl_reward": yl_reward, "train/lr": lr}, step=step)
        
    def log_val(self, step: int, val_loss: float):
        PURPLE = "\033[95m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        
        log_str = f"✨ {BOLD}{PURPLE}[Validation @ Step {step:05d}]{RESET} ➔ {PURPLE}Val Loss:{RESET} {val_loss:.4f} 🎯"
        tqdm.write(log_str)

        if self.use_wandb:
            wandb.log({"val/loss": val_loss}, step=step)
    
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