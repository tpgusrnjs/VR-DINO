import torch
import torch.nn.functional as F


def entropy_from_logits(logits):
    """Calculate entropy to measure prediction uncertainty."""
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return entropy


def consistency_score(global_logits, local_logits):
    """Measure consistency between averaged global and local teacher views."""
    g = F.normalize(global_logits, dim=-1)
    l = F.normalize(local_logits, dim=-1)
    sim = (g * l).sum(dim=-1)
    return sim


class VRKDLoss(torch.nn.Module):
    """VR-KD: Confidence-based Knowledge Distillation Loss"""

    def __init__(self, temperature=0.4, alpha=0.5, min_weight=0.1, center_momentum=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.min_weight = min_weight
        self.center_momentum = center_momentum
        self.center = None

    def forward(self, student_logits, teacher_global_logits,
                teacher_local_logits, epoch_ratio, weight_mode='reliability'):
        """
        Args:
            student_logits: Student predictions on local crops [B, L, C]
            teacher_global_logits: Teacher predictions on global crops [B, G, C]
            teacher_local_logits: Teacher predictions on local crops [B, L, C]
            epoch_ratio: Current epoch / total epochs (for adaptive scheduling)
            weight_mode: 'reliability', 'random', or 'uniform'
        """
        B, L, C = student_logits.shape
        G = teacher_global_logits.shape[1]
        T = self.temperature

        if self.center is None or self.center.shape[0] != C:
            self.center = torch.zeros(C, device=student_logits.device, dtype=student_logits.dtype)

        teacher_local_mean = teacher_local_logits.mean(dim=1)
        entropy = entropy_from_logits(teacher_local_mean)
        max_entropy = torch.log(torch.tensor(C, device=entropy.device, dtype=entropy.dtype))
        entropy_norm = 1.0 - (entropy / max_entropy)

        teacher_global_mean = teacher_global_logits.mean(dim=1)
        consistency = consistency_score(teacher_global_mean, teacher_local_mean)
        consistency = (consistency + 1.0) / 2.0

        alpha_t = epoch_ratio
        if weight_mode == 'reliability':
            reliability = alpha_t * entropy_norm + (1.0 - alpha_t) * consistency
        elif weight_mode == 'random':
            reliability = torch.rand_like(entropy_norm)
        else:
            reliability = torch.ones_like(entropy_norm)

        reliability = reliability.detach().clamp(min=self.min_weight, max=1.0)

        teacher_logits = teacher_global_logits.reshape(B * G, C)
        centered_teacher_logits = teacher_logits - self.center
        teacher_probs = F.softmax(centered_teacher_logits / T, dim=-1).view(B, G, C)

        student_log_probs = F.log_softmax(student_logits.reshape(B * L, C) / T, dim=-1).view(B, L, C)

        t = teacher_probs.unsqueeze(1)
        s = student_log_probs.unsqueeze(2)
        kl = F.kl_div(s, t, reduction='none').sum(dim=-1)
        kd_sample_loss = kl.mean(dim=(1, 2))
        kd_loss = (reliability * kd_sample_loss).mean()

        with torch.no_grad():
            batch_center = teacher_logits.detach().mean(dim=0)
            self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)

        stats = {
            'entropy': entropy.mean().item(),
            'consistency': consistency.mean().item(),
            'reliability': reliability.mean().item(),
            'center_norm': self.center.norm().item(),
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
        student_mean_logits = student_logits.mean(dim=1)
        ce = self.ce_loss(student_mean_logits, labels)

        if self.kd_weight == 0.0 or teacher_global_logits is None or teacher_local_logits is None:
            stats = {
                'entropy': 0.0,
                'consistency': 0.0,
                'reliability': 1.0,
                'center_norm': 0.0,
                'ce_loss': ce.item(),
                'kd_loss': 0.0,
            }
            return ce, stats

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
