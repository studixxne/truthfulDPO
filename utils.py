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
        self.use_wandb = getattr(config, "use_wandb", False)
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
        tqdm.write("\n")
        tqdm.write(f"{Color.BOLD}{Color.BLUE}=" * 60)
        tqdm.write(f"EVALUATION RESULT (Total: {total})")
        tqdm.write(f"=" * 60 + Color.RESET)
        tqdm.write(f"{Color.GREEN}CORRECT{Color.RESET}       : {count['CORRECT']}개 ({count['CORRECT']/total*100:.1f}%)")
        tqdm.write(f"{Color.RED}INCORRECT{Color.RESET}     : {count['INCORRECT']}개 ({count['INCORRECT']/total*100:.1f}%)")
        tqdm.write(f"{Color.YELLOW}OVER_REFUSAL{Color.RESET}  : {count['OVER_REFUSAL']}개 ({count['OVER_REFUSAL']/total*100:.1f}%)")
        tqdm.write(f"{Color.BLUE}-" * 60 + Color.RESET)

        if self.use_wandb:
            metrics = {
                "eval/correct_rate": count['CORRECT'] / total,
                "eval/incorrect_rate": count['INCORRECT'] / total,
                "eval/over_refusal_rate": count['OVER_REFUSAL'] / total,
            }
            
            wandb.log(metrics)

    def log_gen_prompts(self, loop: int, num_loops: int, new_prompts: list[dict]):
        categories_list = [p.get("category", "") for p in new_prompts]
        question_list = [p.get("question", "") for p in new_prompts]

        tqdm.write(f"\n{Color.BOLD}[Loop {loop:02d}|{num_loops:02d}]{Color.RESET} Generating {Color.YELLOW}{Color.BOLD}{len(new_prompts)}{Color.RESET} prompts...")
        tqdm.write(f" ➔ Selected Categories: [{', '.join(set(categories_list))}]")
        
        tqdm.write(f" ➔ 📑 {Color.BOLD}Generated Questions:{Color.RESET}")
        for idx, q in enumerate(question_list):
            tqdm.write(f"    {Color.YELLOW}{idx+1:02d}.{Color.RESET} {q}")
        
        tqdm.write(f"{Color.BLUE}-" * 60 + Color.RESET)

    def log_gen_summary(self, total_prompts: int, file_path: str):
        print("\n")
        print(f"{Color.BOLD}{Color.GREEN}=" * 65)
        print(f"SYNTHETIC PROMPTS GENERATION RESULTS")
        print(f"{Color.GREEN}=" * 65 + Color.RESET)
        print(f"{Color.BOLD}Total Generated{Color.RESET} : {Color.GREEN}{Color.BOLD}{total_prompts}{Color.RESET} prompts")
        print(f"{Color.BOLD}Saved Location{Color.RESET}  : {Color.CYAN}{file_path}{Color.RESET}")
        print(f"{Color.GREEN}-" * 65 + Color.RESET + "\n")

    def log_inference_header(self, mode: str, total: int, model_name: str):
        tqdm.write("\n" + f"{Color.BOLD}{Color.CYAN}=" * 65)
        tqdm.write(f"Model Inference Started [{mode.upper()}]")
        tqdm.write(f"Model Name : {model_name}")
        tqdm.write(f"Total Task : {total} questions")
        tqdm.write(f"{Color.CYAN}=" * 65 + Color.RESET)

    def log_inference_step(self, idx: int, total: int, question: str, mode: str, responses: list[str]):
        tqdm.write(f"{Color.BOLD}[{idx:04d}|{total:04d}] Inferencing{Color.RESET}")
        tqdm.write(f" {Color.CYAN}• Q:{Color.RESET} {question}")

        max_len = 150
        
        if mode == 'eval':
            tqdm.write(f" {Color.GREEN}• Answer:{Color.RESET} {responses[0] if len(responses[0]) < max_len else responses[0][:max_len] + '...'}")
            
        elif mode == 'sample':
            tqdm.write(f"\n {Color.PURPLE}• Response A{Color.RESET}")
            tqdm.write(f"   └ \"{responses[0] if len(responses[0]) < max_len else responses[0][:max_len] + '...'}\"\n")
            tqdm.write(f" {Color.BLUE}• Response B{Color.RESET}")
            tqdm.write(f"   └ \"{responses[1] if len(responses[1]) < max_len else responses[1][:max_len] + '...'}\"")
            
        tqdm.write(f"{Color.CYAN}-" * 60 + Color.RESET)

    def log_inference_summary(self, total: int, file_path: str):
        tqdm.write("\n")
        tqdm.write(f"{Color.BOLD}{Color.GREEN}=" * 65)
        tqdm.write(f"INFERENCE COMPLETE")
        tqdm.write(f"{Color.GREEN}=" * 65 + Color.RESET)
        tqdm.write(f"{Color.BOLD}Total Processed{Color.RESET} : {Color.GREEN}{Color.BOLD}{total}{Color.RESET} prompts")
        tqdm.write(f"{Color.BOLD}Saved Location{Color.RESET}  : {Color.CYAN}{file_path}{Color.RESET}")
        tqdm.write(f"{Color.GREEN}-" * 65 + Color.RESET + "\n")

    def log_mc2_step(self, step: int, total: int, question: str, mc2_score: float):
        tqdm.write(f"{Color.BOLD}[{step+1:04d}|{total:04d}] MC2 Evaluation{Color.RESET}")
        tqdm.write(f" {Color.CYAN}• Q:{Color.RESET} {question}")
        tqdm.write(f" {Color.PURPLE}• MC2 Score:{Color.RESET} {mc2_score:.4f}")
        tqdm.write(f"{Color.CYAN}-" * 60 + Color.RESET)

    def log_mc2_summary(self, total: int, final_score: float):
        tqdm.write("\n")
        tqdm.write(f"{Color.BOLD}{Color.GREEN}=" * 65)
        tqdm.write(f"MC2 EVALUATION COMPLETE")
        tqdm.write(f"{Color.GREEN}=" * 65 + Color.RESET)
        tqdm.write(f"{Color.BOLD}Total Processed{Color.RESET} : {Color.GREEN}{Color.BOLD}{total}{Color.RESET} questions")
        tqdm.write(f"{Color.BOLD}Final MC2 Score{Color.RESET} : {Color.PURPLE}{Color.BOLD}{final_score:.2f}%{Color.RESET}")
        tqdm.write(f"{Color.GREEN}-" * 65 + Color.RESET + "\n")

        if self.use_wandb:
            wandb.log({"eval/mc2_accuracy": final_score})
    
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