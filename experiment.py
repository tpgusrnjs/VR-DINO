"""
VR-DINO Experiment Runner
========================

Comprehensive experiment framework for training and evaluating VR-KD.
Combines training, evaluation, and visualization in unified pipelines.

USAGE:
------
1. Train single configuration (100 epochs):
   python main.py train --config vr_dino

2. Run all 4 experiments (Vanilla, CE-only, Random, VR-DINO):
   python main.py run --epochs 100

3. Train with visualizations:
   python main.py viz --config vr_dino --epochs 100

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
"""

import torch
import argparse
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import torch.cuda.amp as amp
import torch.amp

from models import build_teacher, build_student
from losses import CombinedLoss
from datasets import get_cifar100_loaders
from utils import (
    get_device, set_seed, accuracy, get_baseline_configs,
    AverageMeter, load_checkpoint, save_checkpoint
)
from ablation import generate_ablation_report
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
        
        teacher_global = teacher(global_views.mean(dim=1))
        teacher_local_list = []
        for i in range(local_views.size(1)):
            teacher_local_list.append(teacher(local_views[:, i]))
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
    teacher = build_teacher(num_classes=100).to(device)
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
        num_workers=min(8, torch.cuda.device_count() * 2) if torch.cuda.is_available() else 2
    )
    
    loss_meter = AverageMeter()
    train_loss_history = []
    val_accuracy_history = []
    best_acc = 0
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'{config_name}_best.pth'
    result_path = checkpoint_dir / 'results.json'
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    if checkpoint_path.exists():
        print(f"\nFound existing checkpoint for '{config_name}' at {checkpoint_path}. Skipping training.")
        start_epoch = load_checkpoint(str(checkpoint_path), student)
        if result_path.exists():
            with open(result_path, 'r') as f:
                saved_result = json.load(f)
            saved_result['checkpoint'] = str(checkpoint_path)
            return saved_result
        else:
            # Evaluate if saved result is missing
            test_acc1, test_acc5 = evaluate(student, teacher, test_loader, epochs-1, epochs, device, config.weight_mode)
            return {
                'name': config.name,
                'val_acc': best_acc,
                'test_acc@1': test_acc1,
                'test_acc@5': test_acc5,
                'checkpoint': str(checkpoint_path),
                'history': {
                    'train_loss': train_loss_history,
                    'val_accuracy': val_accuracy_history
                }
            }
    
    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"Weight Mode: {config.weight_mode}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        student.train()
        loss_meter.reset()
        epoch_ratio = epoch / max(epochs, 1)
        
        train_iter = tqdm(
            train_loader,
            desc=f"{config.name} Epoch {epoch+1}/{epochs}",
            leave=False,
            unit='batch'
        )
        
        for (global_views, local_views), labels in train_iter:
            global_views = global_views.to(device)
            local_views = local_views.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                teacher_global = teacher(global_views.mean(dim=1))
                teacher_local_list = []
                for i in range(local_views.size(1)):
                    teacher_local_list.append(teacher(local_views[:, i]))
                teacher_local = torch.stack(teacher_local_list, dim=1).mean(dim=1)
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                student_logits = []
                for i in range(local_views.size(1)):
                    student_logits.append(student(local_views[:, i]))
                student_logits = torch.stack(student_logits, dim=1).mean(dim=1)
                
                loss, _ = criterion(
                    student_logits, teacher_global, teacher_local,
                    labels, epoch_ratio, weight_mode=config.weight_mode
                )
            
            optimizer.zero_grad()
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loss_meter.update(loss.item(), n=labels.size(0))
            
            # Display current validation accuracy if available
            val_display = f"{val_accuracy_history[-1]:.2f}%" if val_accuracy_history else "pending"
            train_iter.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                val=val_display
            )
        
        train_loss_history.append(loss_meter.avg)
        val_acc1, val_acc5 = evaluate(student, teacher, val_loader, epoch, epochs, device, config.weight_mode)
        val_accuracy_history.append(val_acc1)
        
        if val_acc1 > best_acc:
            best_acc = val_acc1
            save_checkpoint({'model_state_dict': student.state_dict()}, str(checkpoint_path))
        
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={loss_meter.avg:.4f}, Val Acc={val_acc1:.2f}%")
    
    # Test
    torch.cuda.empty_cache()
    load_checkpoint(str(checkpoint_path), student)
    test_acc1, test_acc5 = evaluate(student, teacher, test_loader, epochs-1, epochs, device, config.weight_mode)
    
    result = {
        'name': config.name,
        'val_acc': best_acc,
        'test_acc@1': test_acc1,
        'test_acc@5': test_acc5,
        'checkpoint': str(checkpoint_path),
        'history': {
            'train_loss': train_loss_history,
            'val_accuracy': val_accuracy_history
        }
    }
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def collect_model_predictions(student, test_loader, device):
    """Collect predictions and sample images for visualization."""
    student.eval()
    all_predictions = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
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
    
    return torch.cat(all_predictions, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_images, dim=0)


def generate_visualizations(config_name, checkpoint_path, output_dir='./results'):
    """Generate attention and prediction visualizations for a checkpoint."""
    device = get_device()
    result_dir = Path(output_dir) / config_name
    result_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = result_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    teacher = build_teacher(num_classes=100).to(device)
    student = build_student(num_classes=100).to(device)
    student.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    student.eval()
    
    _, _, test_loader = get_cifar100_loaders(batch_size=64, num_workers=4)
    all_predictions, all_labels, all_images = collect_model_predictions(student, test_loader, device)
    
    print(f"\nGenerating visualizations for {config_name}...")
    print("  Extracting attention maps...")
    visualizer = DINoAttentionVisualizer(teacher.model, device=device)
    for i in range(min(4, len(all_images))):
        visualizer.visualize_attention(
            all_images[i],
            output_path=str(viz_dir / f'attention_map_{i}.png')
        )
    visualizer.remove_hooks()
    
    print("  Generating confusion matrix...")
    PredictionVisualizer.confusion_matrix(
        all_predictions, all_labels, num_classes=100,
        output_path=str(viz_dir / 'confusion_matrix.png')
    )
    
    print("  Visualizing predictions...")
    PredictionVisualizer.visualize_predictions(
        all_images[:16], all_predictions[:16], all_labels[:16],
        class_names=None,
        num_samples=16,
        output_path=str(viz_dir / 'predictions.png')
    )


def run_all_experiments(output_dir='./results', epochs=200):
    """Run all 4 experimental configurations"""
    
    result_dir = Path(output_dir)
    result_dir.mkdir(exist_ok=True)
    
    configs = ['vanilla', 'ce_only', 'random', 'vr_dino']
    results = {}
    loss_histories = {}
    
    for config in configs:
        config_dir = result_dir / config
        config_dir.mkdir(parents=True, exist_ok=True)
        result = run_single_experiment(config, checkpoint_dir=str(config_dir), epochs=epochs)
        loss_histories[config] = result.get('history', {}).get('train_loss', [])
        results[config] = result
        
        # Save per-config results and visualizations automatically
        with open(config_dir / 'results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        generate_visualizations(config, checkpoint_path=result['checkpoint'], output_dir=str(result_dir))
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'results': results
    }
    
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate a single comparison plot for training loss
    MetricsVisualizer.plot_loss_comparison(loss_histories, output_dir=str(result_dir))
    
    # Generate ablation plan and save to the results folder
    ablation_dir = result_dir / 'ablations'
    generate_ablation_report(ablation_dir=str(ablation_dir))
    
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
    
    result_dir = Path(output_dir) / config_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result = run_single_experiment(config_name, checkpoint_dir=str(result_dir), epochs=epochs)
    checkpoint_path = result['checkpoint']
    
    generate_visualizations(config_name, checkpoint_path=checkpoint_path, output_dir=output_dir)
    
    with open(result_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment runner')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    train_parser = subparsers.add_parser('train', help='Train single config')
    train_parser.add_argument('--config', required=True,
                             choices=['vanilla', 'ce_only', 'random', 'vr_dino'])
    train_parser.add_argument('--epochs', default=100, type=int)
    train_parser.add_argument('--output-dir', default='./results')
    
    exp_parser = subparsers.add_parser('experiment', help='Run all experiments')
    exp_parser.add_argument('--epochs', default=100, type=int)
    exp_parser.add_argument('--output-dir', default='./results')
    
    viz_parser = subparsers.add_parser('visualize', help='Train and visualize')
    viz_parser.add_argument('--config', required=True,
                           choices=['vanilla', 'ce_only', 'random', 'vr_dino'])
    viz_parser.add_argument('--epochs', default=100, type=int)
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
