import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set up directories
Path("method_boxplots").mkdir(exist_ok=True)

def extract_metrics(filepath):
    """Extracts MAE and R² from log files"""
    metrics = {'MAE': None, 'R2': None}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(
                r"training terminated at epoch \d+\s*-- RESULT.*?"
                r"--- MAE:\s*([\d.]+).*?R2:\s*([\d.-]+)",
                content,
                re.DOTALL
            )
            if match:
                metrics['MAE'] = float(match.group(1))
                metrics['R2'] = float(match.group(2))
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
    return metrics

def collect_method_data(feature_count):
    """Collects data for all methods of a given feature count"""
    data = {
        'PCA': {'MAE': [], 'R2': []},
        'ISOMAP': {'MAE': [], 'R2': []},
        't-SNE': {'MAE': [], 'R2': []},
        'UMAP': {'MAE': [], 'R2': []}
    }
    
    for exp_num in range(1, 4):
        if feature_count == 80:
            method_files = {
                'PCA': f'logs/finetune_pca_80features_800epoch_{exp_num}.log',
                'ISOMAP': f'logs/finetune_isomap_80features_800epoch_{exp_num}.log',
                't-SNE': f'logs/finetune_tsne_3features_800epoch_{exp_num}.log',
                'UMAP': f'logs/finetune_umap_80features_800epoch_{exp_num}.log'
            }
        else:
            method_files = {
                'PCA': f'logs/finetune_pca_40features_{exp_num}.log',
                'ISOMAP': f'logs/finetune_isomap_40features_{exp_num}.log',
                't-SNE': f'logs/finetune_tsne_3features_{exp_num}.log',
                'UMAP': f'logs/finetune_umap_40features_{exp_num}.log'
            }
        
        for method, filepath in method_files.items():
            if os.path.exists(filepath):
                metrics = extract_metrics(filepath)
                if metrics['MAE'] is not None:
                    data[method]['MAE'].append(metrics['MAE'])
                    data[method]['R2'].append(metrics['R2'])
    
    return data

def create_method_boxplots():
    """Creates boxplots comparing methods across dimensions"""
    # Collect data for both feature counts
    data_40 = collect_method_data(40)
    data_80 = collect_method_data(80)
    
    # Determine global y-axis limits for MAE
    all_mae_values = []
    for method in ['PCA', 'ISOMAP', 't-SNE', 'UMAP']:
        all_mae_values.extend(data_40[method]['MAE'])
        all_mae_values.extend(data_80[method]['MAE'])
    mae_min, mae_max = min(all_mae_values), max(all_mae_values)
    mae_padding = (mae_max - mae_min) * 0.1  # 10% padding
    
    # Create MAE comparison plots
    plt.figure(figsize=(12, 6))
    
    # 40D MAE
    plt.subplot(1, 2, 1)
    plt.boxplot([data_40[method]['MAE'] for method in ['PCA', 'ISOMAP', 't-SNE', 'UMAP']],
                labels=['PCA', 'ISOMAP', 't-SNE', 'UMAP'])
    plt.title('MAE Comparison (40 Dimensions)')
    plt.ylabel('MAE')
    plt.ylim(mae_min - mae_padding, mae_max + mae_padding)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 80D MAE
    plt.subplot(1, 2, 2)
    plt.boxplot([data_80[method]['MAE'] for method in ['PCA', 'ISOMAP', 't-SNE', 'UMAP']],
                labels=['PCA', 'ISOMAP', 't-SNE', 'UMAP'])
    plt.title('MAE Comparison (80 Dimensions)')
    plt.ylabel('MAE')
    plt.ylim(mae_min - mae_padding, mae_max + mae_padding)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('method_boxplots/mae_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determine global y-axis limits for R²
    all_r2_values = []
    for method in ['PCA', 'ISOMAP', 't-SNE', 'UMAP']:
        all_r2_values.extend(data_40[method]['R2'])
        all_r2_values.extend(data_80[method]['R2'])
    r2_min, r2_max = min(all_r2_values), max(all_r2_values)
    r2_padding = (r2_max - r2_min) * 0.1  # 10% padding
    
    # Create R² comparison plots
    plt.figure(figsize=(12, 6))
    
    # 40D R²
    plt.subplot(1, 2, 1)
    plt.boxplot([data_40[method]['R2'] for method in ['PCA', 'ISOMAP', 't-SNE', 'UMAP']],
                labels=['PCA', 'ISOMAP', 't-SNE', 'UMAP'])
    plt.title('R² Comparison (40 Dimensions)')
    plt.ylabel('R² Score')
    plt.ylim(r2_min - r2_padding, r2_max + r2_padding)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 80D R²
    plt.subplot(1, 2, 2)
    plt.boxplot([data_80[method]['R2'] for method in ['PCA', 'ISOMAP', 't-SNE', 'UMAP']],
                labels=['PCA', 'ISOMAP', 't-SNE', 'UMAP'])
    plt.title('R² Comparison (80 Dimensions)')
    plt.ylabel('R² Score')
    plt.ylim(r2_min - r2_padding, r2_max + r2_padding)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('method_boxplots/r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_method_boxplots()
    print("Method comparison boxplots created and saved in the 'method_boxplots' directory:")
    print("- mae_comparison.png (MAE for 40D and 80D)")
    print("- r2_comparison.png (R² for 40D and 80D)")