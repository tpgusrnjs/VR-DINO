import torch
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from utils import get_baseline_configs


@dataclass
class AblationConfig:
    """Configuration for ablation studies"""
    param_name: str
    param_values: list
    base_config: str = 'vr_dino'


def ablation_temperature():
    """Test different temperature values: [0.1, 0.3, 0.5, 0.7, 1.0]"""
    temps = [0.1, 0.3, 0.5, 0.7, 1.0]
    output_dir = './ablations/temperature'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for temp in temps:
        print(f"\n{'='*80}")
        print(f"Testing Temperature: {temp}")
        print(f"{'='*80}\n")
        
        config = get_baseline_configs()['vr_dino']
        config.temperature = temp
        
        # Modify train to accept config object (would need refactor)
        # For now, just log the configuration
        results[temp] = {
            'temperature': temp,
            'description': f'Knowledge distillation temperature'
        }
    
    with open(f'{output_dir}/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def ablation_alpha():
    """Test different alpha values: [0.0, 0.25, 0.5, 0.75, 1.0]"""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    output_dir = './ablations/alpha'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for alpha in alphas:
        print(f"\n{'='*80}")
        print(f"Testing Alpha: {alpha}")
        print(f"Alpha = 0.0: Pure consistency weighting (all epochs)")
        print(f"Alpha = 1.0: Pure entropy weighting (all epochs)")
        print(f"Alpha = 0.5: Balanced (adaptive scheduling)")
        print(f"{'='*80}\n")
        
        config = get_baseline_configs()['vr_dino']
        config.alpha = alpha
        
        results[alpha] = {
            'alpha': alpha,
            'description': f'Entropy vs Consistency weighting ratio'
        }
    
    with open(f'{output_dir}/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def ablation_min_weight():
    """Test different minimum weight thresholds: [0.0, 0.05, 0.1, 0.2, 0.5]"""
    min_weights = [0.0, 0.05, 0.1, 0.2, 0.5]
    output_dir = './ablations/min_weight'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for min_w in min_weights:
        print(f"\n{'='*80}")
        print(f"Testing Min Weight: {min_w}")
        print(f"Prevents collapse: weights clipped to [{min_w}, 1.0]")
        print(f"{'='*80}\n")
        
        config = get_baseline_configs()['vr_dino']
        config.min_weight = min_w
        
        results[min_w] = {
            'min_weight': min_w,
            'description': f'Minimum reliability weight threshold'
        }
    
    with open(f'{output_dir}/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def generate_ablation_report(ablation_dir='./ablations'):
    """Generate comprehensive ablation study report"""
    Path(ablation_dir).mkdir(exist_ok=True)
    
    report = {
        'title': 'VR-DINO Ablation Study',
        'studies': {
            'temperature': {
                'description': 'Knowledge distillation temperature (higher = softer targets)',
                'hypothesis': 'Moderate values (0.3-0.5) should balance knowledge transfer',
                'parameter_range': [0.1, 0.3, 0.5, 0.7, 1.0]
            },
            'alpha': {
                'description': 'Initial entropy-consistency weighting ratio',
                'hypothesis': 'Balanced (0.5) with adaptive scheduling should work best',
                'variants': {
                    '0.0': 'Pure consistency (multiview stable)',
                    '0.5': 'Balanced with linear scheduling',
                    '1.0': 'Pure entropy (confident predictions)'
                }
            },
            'min_weight': {
                'description': 'Minimum reliability weight threshold',
                'hypothesis': '0.1 prevents collapse while maintaining weighting benefit',
                'parameter_range': [0.0, 0.05, 0.1, 0.2, 0.5]
            },
        },
        'interpretation_guide': {
            'temperature': {
                'low (0.1)': 'Sharp targets, high KL divergence, harder to match',
                'high (1.0)': 'Soft targets, low KL divergence, easier to match'
            },
            'alpha_adaptive': 'Early (α↓): Consistency focus → Late (α↑): Entropy focus',
            'min_weight_0.0': 'Risks zero-weight instability',
            'min_weight_0.1': 'Balances selectivity and stability'
        }
    }
    
    with open(f'{ablation_dir}/ablation_plan.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("ABLATION STUDY PLAN")
    print("="*80 + "\n")
    
    for study, details in report['studies'].items():
        print(f"\n{study.upper()}")
        print(f"  Description: {details['description']}")
        print(f"  Hypothesis: {details['hypothesis']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation Studies for VR-DINO')
    parser.add_argument('--study', choices=['all', 'temperature', 'alpha', 'min_weight'],
                       default='all', help='Which ablation study to run')
    parser.add_argument('--plan-only', action='store_true', help='Only generate ablation plan')
    
    args = parser.parse_args()
    
    if args.plan_only or args.study == 'all':
        generate_ablation_report()
    
    if args.study == 'temperature':
        ablation_temperature()
    elif args.study == 'alpha':
        ablation_alpha()
    elif args.study == 'min_weight':
        ablation_min_weight()
    elif args.study == 'all':
        print("Running all ablation studies (plan-only mode by default)")
        print("To actually run them, modify the config and run train.py separately")
