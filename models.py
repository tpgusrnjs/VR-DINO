import torch
import torch.nn as nn
import timm


class TeacherModel(nn.Module):
    """Frozen DINO teacher for feature extraction"""
    
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits16'
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)


class StudentModel(nn.Module):
    """Student ViT for knowledge distillation"""
    
    def __init__(self, num_classes=100, model_name='vit_tiny_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0
        )
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def build_teacher():
    """Load pretrained DINO teacher model"""
    return TeacherModel()


def build_student(num_classes=100, model_name='vit_tiny_patch16_224'):
    """Build student model for knowledge distillation"""
    return StudentModel(num_classes=num_classes, model_name=model_name)