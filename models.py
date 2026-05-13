import torch
import torch.nn as nn
import timm


class StudentModel(nn.Module):
    """Student ViT for knowledge distillation"""

    def __init__(self, num_classes=100, model_name='vit_tiny_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=224,
            dynamic_img_size=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.model = self.backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class TeacherModel(StudentModel):
    """Momentum teacher network for DINO-style distillation"""

    def __init__(self, num_classes=100, model_name='vit_tiny_patch16_224'):
        super().__init__(num_classes=num_classes, model_name=model_name)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self, student, momentum=0.996):
        for teacher_param, student_param in zip(self.parameters(), student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buffer, student_buffer in zip(self.buffers(), student.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)


def build_teacher(num_classes=100, model_name='vit_tiny_patch16_224'):
    """Build a momentum teacher network with the same architecture as the student."""
    return TeacherModel(num_classes=num_classes, model_name=model_name)


def build_student(num_classes=100, model_name='vit_tiny_patch16_224'):
    """Build student model for knowledge distillation"""
    return StudentModel(num_classes=num_classes, model_name=model_name)
