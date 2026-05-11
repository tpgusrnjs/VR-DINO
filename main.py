"""
VR-DINO Main Entry Point
========================

One-stop interface for all experiment operations.

QUICK START:
-----------
1. Train single method (100 epochs):
   python main.py train --config vr_dino

2. Full experiment (4 methods, 100 epochs + ablation + visualizations):
   python main.py run

3. Custom epochs:
   python main.py run --epochs 200

RESULTS LOCATION:
-----------------
All results → results/{config}/
├─ results.json
├─ {config}_best.pth
└─ visualizations/
    ├─ attention_map_*.png
    ├─ confusion_matrix.png
    └─ predictions.png

Loss Comparison:
├─ loss_comparison.png    # All 4 methods on one graph

EXPECTED TIME:
--------------
train       : 2-4 hours per config
run         : 8-16 hours (all 4 configs + ablations + viz)
"""

import argparse
from experiment import run_single_experiment, run_all_experiments


def main():
    parser = argparse.ArgumentParser(description='VR-DINO: Knowledge Distillation Framework')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train single config
    train_parser = subparsers.add_parser('train', help='Train single config')
    train_parser.add_argument('--config', required=True,
                             choices=['vanilla', 'ce_only', 'random', 'vr_dino'])
    train_parser.add_argument('--epochs', default=100, type=int)
    train_parser.add_argument('--output-dir', default='./results')
    
    # Run all experiments with visualizations
    exp_parser = subparsers.add_parser('run', help='Run all 4 experiments with visualizations')
    exp_parser.add_argument('--epochs', default=100, type=int)
    exp_parser.add_argument('--output-dir', default='./results')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("\n" + "="*70)
        print(f"Training: {args.config} ({args.epochs} epochs)")
        print("="*70)
        run_single_experiment(args.config, checkpoint_dir=args.output_dir, epochs=args.epochs)
    
    elif args.command == 'run':
        print("\n" + "="*70)
        print(f"Running Full Experiment Suite ({args.epochs} epochs)")
        print("Including: Training + Ablations + Visualizations")
        print("="*70)
        run_all_experiments(output_dir=args.output_dir, epochs=args.epochs)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 