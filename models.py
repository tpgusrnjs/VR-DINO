import torch
import torch.nn as nn
import timm


class TeacherModel(nn.Module):
    """Frozen DINO teacher for feature extraction"""
    
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits16'
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Add classification head for KD
        self.head = nn.Linear(self.model.embed_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            logits = self.head(features)
            return logits


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


def build_teacher(num_classes=100):
    """Load pretrained DINO teacher model"""
    return TeacherModel(num_classes=num_classes)


def build_student(num_classes=100, model_name='vit_tiny_patch16_224'):
    """Build student model for knowledge distillation"""
    return StudentModel(num_classes=num_classes, model_name=model_name)