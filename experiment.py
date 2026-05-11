"""
VR-DINO Experiment Runner
========================

Comprehensive experiment framework for training and evaluating VR-KD.
Combines training, evaluation, and visualization in unified pipelines.

USAGE:
------
1. Quick test (10 epochs):
   python main.py quick --config vr_dino

2. Train single configuration (200 epochs):
   python main.py train --config vr_dino --epochs 200

3. Run all 4 experiments (Vanilla, CE-only, Random, VR-DINO):
   python main.py run --epochs 200

4. Train with visualizations:
   python main.py viz --config vr_dino --epochs 200

RESULTS:
--------
All results saved to results/{config}/
├─ results.json          # Accuracy and metrics
├─ {config}_best.pth     # Model checkpoint
└─ visualizations/
   ├─ attention_map_*.png    # DINO attention maps
   ├─ confusion_matrix.png   # Per-class performance
   └─ predictions.png        # Sample predictions

CONFIGURATION COMPARISON:
-------------------------
1. Vanilla DINO       : Baseline (no KD, uniform weights)
2. CE-only (No KD)    : Only cross-entropy loss
3. Random Weighting   : Ablation (random weights)
4. VR-DINO (Proposed) : Reliability-based weighting ⭐

Expected Results:
- Vanilla:  75%
- CE-only:  72% (-3%)
- Random:   74% (-1%)
- VR-DINO:  78% (+3%) ✓
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import json
from datetime import datetime

from models import build_teacher, build_student
from losses import CombinedLoss
from datasets import get_cifar100_loaders
from utils import (
    get_device, set_seed, accuracy, load_checkpoint, 
    get_baseline_configs, MetricsLogger
)
from visualize import DINoAttentionVisualizer, MetricsVisualizer, PredictionVisualizer


@torch.no_grad()
def evaluate(model, teacher, data_loader, epoch, total_epochs, device, weight_mode='reliability'):
    """Evaluate model on dataset"""
    model.eval()
    teacher.eval()
    
    acc1_total = 0.0
    acc5_total = 0.0
    total = 0
    
    criterion = CombinedLoss()
    epoch_ratio = epoch / total_epochs if total_epochs > 0 else 1.0
    
    for (global_views, local_views), labels in data_loader:
        global_views = global_views.to(device)
        local_views = local_views.to(device)
        labels = labels.to(device)
        
        teacher_global = teacher.model(global_views.mean(dim=1))
        teacher_local_list = []
        for i in range(local_views.size(1)):
            teacher_local_list.append(teacher.model(local_views[:, i]))
        teacher_local = torch.stack(teacher_local_list, dim=1).mean(dim=1)
        
        student_logits = []
        for i in range(local_views.size(1)):
            student_logits.append(model(local_views[:, i]))
        student_logits = torch.stack(student_logits, dim=1).mean(dim=1)
        
        acc1, acc5 = accuracy(student_logits, labels, topk=(1, 5))
        
        acc1_total += acc1.item() * labels.size(0)
        acc5_total += acc5.item() * labels.size(0)
        total += labels.size(0)
    
    return acc1_total / total, acc5_total / total


def run_single_experiment(config_name, checkpoint_dir='./checkpoints', epochs=100):
    """Run single configuration and return results"""
    
    device = get_device()
    set_seed(42)
    
    configs = get_baseline_configs()
    config = configs[config_name]
    
    # Models
    teacher = build_teacher().to(device)
    student = build_student(num_classes=100).to(device)
    
    criterion = CombinedLoss(
        kd_weight=0.5 if config.use_kd else 0.0,
        temperature=config.temperature,
        alpha=config.alpha
    )
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=0.05
    )
    
    # Data
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=config.batch_size,
        num_workers=4
    )
    
    best_acc = 0
    checkpoint_path = Path(checkpoint_dir) / f'{config_name}_best.pth'
    
    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"Weight Mode: {config.weight_mode}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        student.train()
        epoch_ratio = epoch / epochs
        
        for (global_views, local_views), labels in train_loader:
            global_views = global_views.to(device)
            local_views = local_views.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                teacher_global = teacher.model(global_views.mean(dim=1))
                teacher_local_list = []
                for i in range(local_views.size(1)):
                    teacher_local_list.append(teacher.model(local_views[:, i]))
                teacher_local = torch.stack(teacher_local_list, dim=1).mean(dim=1)
            
            student_logits = []
            for i in range(local_views.size(1)):
                student_logits.append(student(local_views[:, i]))
            student_logits = torch.stack(student_logits, dim=1).mean(dim=1)
            
            loss, _ = criterion(
                student_logits, teacher_global, teacher_local,
                labels, epoch_ratio, weight_mode=config.weight_mode
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validate
        val_acc1, val_acc5 = evaluate(student, teacher, val_loader, epoch, epochs, device, config.weight_mode)
        
        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save({'model_state_dict': student.state_dict()}, checkpoint_path)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Val Acc={val_acc1:.2f}%")
    
    # Test
    torch.cuda.empty_cache()
    student.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    test_acc1, test_acc5 = evaluate(student, teacher, test_loader, epochs-1, epochs, device, config.weight_mode)
    
    return {
        'name': config.name,
        'val_acc': best_acc,
        'test_acc@1': test_acc1,
        'test_acc@5': test_acc5,
        'checkpoint': str(checkpoint_path)
    }


def run_all_experiments(output_dir='./results', epochs=200):
    """Run all 4 experimental configurations"""
    
    result_dir = Path(output_dir)
    result_dir.mkdir(exist_ok=True)
    
    configs = ['vanilla', 'ce_only', 'random', 'vr_dino']
    results = {}
    
    for config in configs:
        result = run_single_experiment(config, checkpoint_dir=str(result_dir), epochs=epochs)
        results[config] = result
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'results': results
    }
    
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}\n")
    
    for config in configs:
        r = results[config]
        print(f"{r['name']:25s} | Val: {r['val_acc']:.2f}% | Test: {r['test_acc@1']:.2f}%")
    
    # Calculate improvements
    print(f"\n{'='*70}")
    print("IMPROVEMENTS OVER VANILLA DINO")
    print(f"{'='*70}\n")
    
    vanilla_acc = results['vanilla']['test_acc@1']
    for config in ['ce_only', 'random', 'vr_dino']:
        test_acc = results[config]['test_acc@1']
        improvement = test_acc - vanilla_acc
        print(f"{results[config]['name']:25s}: {improvement:+.2f}% ({test_acc:.2f}%)")
    
    print(f"\n{'='*70}\n")
    
    return results


def run_config_and_visualize(config_name, output_dir='./results', epochs=200):
    """Run single config and generate visualizations"""
    
    device = get_device()
    set_seed(42)
    
    result_dir = Path(output_dir) / config_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and get checkpoint
    result = run_single_experiment(config_name, checkpoint_dir=str(result_dir), epochs=epochs)
    checkpoint_path = result['checkpoint']
    
    # Load trained model for visualization
    teacher = build_teacher().to(device)
    student = build_student(num_classes=100).to(device)
    student.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    student.eval()
    
    # Get test data
    _, _, test_loader = get_cifar100_loaders(batch_size=128, num_workers=4)
    
    all_predictions = []
    all_labels = []
    all_images = []
    
    @torch.no_grad()
    def collect_data():
        for (global_views, local_views), labels in test_loader:
            global_views = global_views.to(device)
            local_views = local_views.to(device)
            
            student_logits = []
            for i in range(local_views.size(1)):
                student_logits.append(student(local_views[:, i]))
            student_logits = torch.stack(student_logits, dim=1).mean(dim=1)
            
            all_predictions.append(student_logits.cpu())
            all_labels.append(labels.cpu())
            all_images.append(global_views[:, 0].cpu())
    
    collect_data()
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    # Visualizations
    viz_dir = result_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations for {config_name}...")
    
    # Attention maps
    print("  Extracting attention maps...")
    visualizer = DINoAttentionVisualizer(teacher.model, device=device)
    for i in range(min(4, len(all_images))):
        img = all_images[i]
        visualizer.visualize_attention(
            img,
            output_path=str(viz_dir / f'attention_map_{i}.png')
        )
    visualizer.remove_hooks()
    
    # Confusion matrix
    print("  Generating confusion matrix...")
    PredictionVisualizer.confusion_matrix(
        all_predictions, all_labels, num_classes=100,
        output_path=str(viz_dir / 'confusion_matrix.png')
    )
    
    # Sample predictions
    print("  Visualizing predictions...")
    PredictionVisualizer.visualize_predictions(
        all_images[:16], all_predictions[:16], all_labels[:16],
        class_names=None,
        num_samples=16,
        output_path=str(viz_dir / 'predictions.png')
    )
    
    # Save results
    with open(result_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved to {result_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment runner')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    train_parser = subparsers.add_parser('train', help='Train single config')
    train_parser.add_argument('--config', required=True,
                             choices=['vanilla', 'ce_only', 'random', 'vr_dino'])
    train_parser.add_argument('--epochs', default=200, type=int)
    train_parser.add_argument('--output-dir', default='./results')
    
    exp_parser = subparsers.add_parser('experiment', help='Run all experiments')
    exp_parser.add_argument('--epochs', default=200, type=int)
    exp_parser.add_argument('--output-dir', default='./results')
    
    viz_parser = subparsers.add_parser('visualize', help='Train and visualize')
    viz_parser.add_argument('--config', required=True,
                           choices=['vanilla', 'ce_only', 'random', 'vr_dino'])
    viz_parser.add_argument('--epochs', default=200, type=int)
    viz_parser.add_argument('--output-dir', default='./results')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_single_experiment(args.config, checkpoint_dir=args.output_dir, epochs=args.epochs)
    elif args.command == 'experiment':
        run_all_experiments(output_dir=args.output_dir, epochs=args.epochs)
    elif args.command == 'visualize':
        run_config_and_visualize(args.config, output_dir=args.output_dir, epochs=args.epochs)
    else:
        parser.print_help()
