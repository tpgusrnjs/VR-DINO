import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from pathlib import Path

from models import build_student
from datasets import StandardTransform
from utils import get_device, load_checkpoint


class ImageInference:
    """Inference wrapper for single image predictions"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model = build_student(num_classes=100).to(device)
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()
        
        self.transform = StandardTransform()
    
    @torch.no_grad()
    def predict(self, image_path, top_k=5):
        """Predict on single image"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)
        
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'class_id': idx.item(),
                'probability': prob.item()
            })
        
        return results
    
    def predict_batch(self, image_dir, top_k=5):
        """Predict on batch of images"""
        image_files = list(Path(image_dir).glob('*.jpg')) + \
                     list(Path(image_dir).glob('*.png'))
        
        all_results = {}
        for image_file in image_files:
            results = self.predict(str(image_file), top_k=top_k)
            all_results[image_file.name] = results
        
        return all_results


def export_model(checkpoint_path, export_path='model.pt'):
    """Export model to TorchScript for deployment"""
    device = get_device()
    model = build_student(num_classes=100).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(export_path)
    
    print(f"Model exported to {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference utilities')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--image', help='Single image for prediction')
    parser.add_argument('--batch', help='Directory with images for batch prediction')
    parser.add_argument('--export', help='Export to TorchScript at path')
    parser.add_argument('--top-k', default=5, type=int, help='Top-K predictions')
    
    args = parser.parse_args()
    
    device = get_device()
    inferencer = ImageInference(args.checkpoint, device=device)
    
    if args.image:
        results = inferencer.predict(args.image, top_k=args.top_k)
        print(f"\nPredictions for {args.image}:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Class {result['class_id']}: {result['probability']*100:.2f}%")
    
    if args.batch:
        results = inferencer.predict_batch(args.batch, top_k=args.top_k)
        print(f"\nBatch predictions for {len(results)} images:")
        for filename, preds in results.items():
            print(f"  {filename}: Class {preds[0]['class_id']} ({preds[0]['probability']*100:.2f}%)")
    
    if args.export:
        export_model(args.checkpoint, args.export)

