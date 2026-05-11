import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import seaborn as sns


class DINoAttentionVisualizer:
    """Extract and visualize DINO self-attention maps"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attentions = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention maps"""
        def create_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(module, 'attn'):
                    attention = module.attn
                    if isinstance(attention, dict):
                        attention = attention.get('attn_probs', None)
                    if attention is not None:
                        self.attentions.append({
                            'layer': layer_idx,
                            'attn': attention.detach().cpu()
                        })
            return hook
        
        for idx, layer in enumerate(self.model.blocks):
            hook = layer.register_forward_hook(create_hook(idx))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def get_attention_map(self, image, layer_idx=11, head_idx=0, threshold=0.0):
        """Extract attention map from specific layer and head"""
        self.attentions = []
        
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0).to(self.device))
        
        if not self.attentions or layer_idx >= len(self.attentions):
            return None
        
        attn = self.attentions[layer_idx]['attn']
        
        if head_idx < attn.shape[1]:
            attn_map = attn[0, head_idx, 0, 1:]
        else:
            attn_map = attn[0, 0, 0, 1:]
        
        num_patches = int(np.sqrt(len(attn_map)))
        attn_map = attn_map.reshape(num_patches, num_patches)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        if threshold > 0:
            attn_map = attn_map * (attn_map > threshold).float()
        
        return attn_map
    
    def visualize_attention(self, image, output_path='attention_map.png', cmap='jet'):
        """Visualize attention map overlaid on image"""
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        attn_map = self.get_attention_map(image, layer_idx=11, threshold=0.1)
        if attn_map is not None:
            axes[1].imshow(attn_map.numpy(), cmap=cmap)
            axes[1].set_title('Attention Map')
            axes[1].axis('off')
            
            image_resized = F.interpolate(
                image.unsqueeze(0).unsqueeze(0).float(),
                size=attn_map.shape,
                mode='bilinear'
            ).squeeze()
            
            axes[2].imshow(image_np, alpha=0.5)
            axes[2].imshow(attn_map.numpy(), cmap=cmap, alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class MetricsVisualizer:
    """Visualize training metrics"""
    
    @staticmethod
    def plot_training_curves(metrics, output_dir='./visualizations'):
        """Plot training loss and metrics"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'loss' in metrics:
            axes[0, 0].plot(metrics['loss'], label='Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        if 'accuracy' in metrics:
            axes[0, 1].plot(metrics['accuracy'], label='Train Acc', linewidth=2)
            if 'val_accuracy' in metrics:
                axes[0, 1].plot(metrics['val_accuracy'], label='Val Acc', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        if 'entropy' in metrics:
            axes[1, 0].plot(metrics['entropy'], label='Entropy', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].set_title('Teacher Entropy (Confidence)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        if 'consistency' in metrics:
            axes[1, 1].plot(metrics['consistency'], label='Consistency', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Consistency Score')
            axes[1, 1].set_title('Global-Local Consistency')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_comparison(results_dict, output_dir='./visualizations'):
        """Compare results across different methods"""
        Path(output_dir).mkdir(exist_ok=True)
        
        methods = list(results_dict.keys())
        accuracies = [results_dict[m]['accuracy'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(methods, accuracies, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(accuracies) * 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/method_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_reliability_distribution(reliabilities, output_path='reliability_dist.png'):
        """Plot distribution of reliability scores"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(reliabilities.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Reliability Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Reliability Score Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].boxplot(reliabilities.cpu().numpy())
        axes[1].set_ylabel('Reliability Score')
        axes[1].set_title('Reliability Score Statistics')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class PredictionVisualizer:
    """Visualize model predictions"""
    
    @staticmethod
    def visualize_predictions(images, predictions, labels, class_names=None, num_samples=8, output_path='predictions.png'):
        """Visualize correct and incorrect predictions"""
        batch_size = min(num_samples, len(images))
        fig, axes = plt.subplots(2, batch_size // 2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(batch_size):
            img = images[i].cpu()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.permute(1, 2, 0).numpy()
            
            pred = predictions[i].cpu().argmax().item()
            label = labels[i].cpu().item()
            
            axes[i].imshow(img)
            
            if class_names is not None and pred < len(class_names):
                pred_name = class_names[pred]
                label_name = class_names[label]
            else:
                pred_name = str(pred)
                label_name = str(label)
            
            title_color = 'green' if pred == label else 'red'
            axes[i].set_title(f'Pred: {pred_name}\nTrue: {label_name}', color=title_color, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def confusion_matrix(predictions, labels, num_classes, output_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix as cm
        
        pred_labels = predictions.argmax(dim=1).cpu().numpy()
        label_np = labels.cpu().numpy()
        
        conf_matrix = cm(label_np, pred_labels, labels=range(num_classes))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
