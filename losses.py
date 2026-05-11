import torch
import torch.nn.functional as F


def entropy_from_logits(logits):
    """Calculate entropy to measure prediction uncertainty"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy


def consistency_score(global_logits, local_logits):
    """Measure consistency between global and local views"""
    g = F.normalize(global_logits, dim=-1)
    l = F.normalize(local_logits, dim=-1)
    sim = (g * l).sum(dim=-1)
    return sim


class VRKDLoss(torch.nn.Module):
    """VR-KD: Confidence-based Knowledge Distillation Loss"""
    
    def __init__(self, temperature=0.4, alpha=0.5, min_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.min_weight = min_weight
    
    def forward(self, student_logits, teacher_global_logits, 
                teacher_local_logits, epoch_ratio, weight_mode='reliability'):
        """
        Args:
            student_logits: Student predictions on local views
            teacher_global_logits: Teacher predictions on global views
            teacher_local_logits: Teacher predictions on local views
            epoch_ratio: Current epoch / total epochs (for adaptive scheduling)
            weight_mode: 'reliability', 'random', or 'uniform'
        """
        T = self.temperature
        
        # Entropy reliability: measure confidence
        entropy = entropy_from_logits(teacher_local_logits)
        max_entropy = torch.log(torch.tensor(
            teacher_local_logits.size(-1),
            device=entropy.device,
            dtype=entropy.dtype
        ))
        entropy_norm = 1.0 - (entropy / max_entropy)
        
        # Consistency reliability: global-local agreement
        consistency = consistency_score(
            teacher_global_logits,
            teacher_local_logits
        )
        consistency = (consistency + 1) / 2
        
        # Adaptive scheduling
        alpha_t = epoch_ratio
        
        if weight_mode == 'reliability':
            # VR-KD: combine entropy and consistency
            reliability = (
                alpha_t * entropy_norm +
                (1 - alpha_t) * consistency
            )
        elif weight_mode == 'random':
            # Random weights for ablation
            reliability = torch.rand_like(entropy_norm)
        else:  # 'uniform'
            # Uniform weights (vanilla baseline)
            reliability = torch.ones_like(entropy_norm)
        
        reliability = reliability.detach()
        reliability = torch.clamp(reliability, min=self.min_weight, max=1.0)
        
        # Knowledge distillation loss
        s = F.log_softmax(student_logits / T, dim=-1)
        t = F.softmax(teacher_local_logits / T, dim=-1)
        
        kd_loss = F.kl_div(s, t, reduction='none').sum(dim=-1)
        kd_loss = (reliability * kd_loss).mean()
        
        stats = {
            'entropy': entropy.mean().item(),
            'consistency': consistency.mean().item(),
            'reliability': reliability.mean().item(),
        }
        
        return kd_loss, stats


class CombinedLoss(torch.nn.Module):
    """Combined CE and KD loss"""
    
    def __init__(self, kd_weight=0.5, temperature=0.4, alpha=0.5):
        super().__init__()
        self.kd_weight = kd_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kd_loss = VRKDLoss(temperature=temperature, alpha=alpha)
    
    def forward(self, student_logits, teacher_global_logits, 
                teacher_local_logits, labels, epoch_ratio, weight_mode='reliability'):
        
        ce = self.ce_loss(student_logits, labels)
        kd, stats = self.kd_loss(
            student_logits,
            teacher_global_logits,
            teacher_local_logits,
            epoch_ratio,
            weight_mode=weight_mode
        )
        
        loss = ce + self.kd_weight * kd
        stats['ce_loss'] = ce.item()
        stats['kd_loss'] = kd.item()
        
        return loss, stats