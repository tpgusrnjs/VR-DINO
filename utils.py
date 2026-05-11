import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for different experimental setups"""
    name: str
    use_kd: bool = True
    use_weighting: bool = True
    weight_mode: str = "reliability"  # reliability, random, uniform
    learning_rate: float = 5e-4
    batch_size: int = 128
    epochs: int = 200
    temperature: float = 0.4
    alpha: float = 0.5
    min_weight: float = 0.1
    warmup_epochs: int = 10
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


def get_baseline_configs():
    """Get predefined baseline configurations"""
    return {
        'vanilla': ExperimentConfig(
            name='Vanilla DINO',
            use_kd=False,
            use_weighting=False,
            weight_mode='uniform'
        ),
        'ce_only': ExperimentConfig(
            name='CE-only (No KD)',
            use_kd=False,
            use_weighting=False,
            weight_mode='uniform'
        ),
        'random': ExperimentConfig(
            name='Random Weighting',
            use_kd=True,
            use_weighting=True,
            weight_mode='random'
        ),
        'vr_dino': ExperimentConfig(
            name='VR-DINO (Proposed)',
            use_kd=True,
            use_weighting=True,
            weight_mode='reliability'
        )
    }


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsLogger:
    def __init__(self, log_dir='./logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.log_dir / timestamp
        self.exp_dir.mkdir(exist_ok=True)
        
        self.metrics = {}
        self.log_file = self.exp_dir / 'metrics.json'
    
    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(val)
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self):
        summary = {}
        for key, vals in self.metrics.items():
            if vals:
                summary[f'{key}_avg'] = np.mean(vals)
                summary[f'{key}_std'] = np.std(vals)
        return summary


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get CUDA device if available"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def init_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def clip_weights(weights, min_val=0.1, max_val=1.0):
    """Clip weights to prevent numerical instability"""
    return torch.clamp(weights, min=min_val, max=max_val)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create cosine annealing schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)))))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def accuracy(output, target, topk=(1, 5)):
    """Compute accuracy@k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filepath):
    """Save training checkpoint"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)
