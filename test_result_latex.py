import os
import re
from pathlib import Path

# Create output directories
Path("latex_test_tables").mkdir(exist_ok=True)

def extract_test_results(filepath):
    """Precisely extracts test results from log files"""
    results = {
        'MAE': None,
        'RMSE': None,
        'R2': None,
        'Spearman': None
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # More accurate pattern that matches your exact log format
            result_match = re.search(
                r"training terminated at epoch \d+\s*-- RESULT.*?"
                r"--- MAE:\s*([\d.]+),\s*RMSE:\s*([\d.]+),\s*R2:\s*([\d.-]+),\s*Spearman:\s*([\d.-]+)",
                content,
                re.DOTALL
            )
            
            if result_match:
                results['MAE'] = float(result_match.group(1))
                results['RMSE'] = float(result_match.group(2))
                results['R2'] = float(result_match.group(3))
                results['Spearman'] = float(result_match.group(4))
            else:
                print(f"Warning: No test results found in {filepath}")
                
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
    
    return results

def generate_test_latex_table(method_data, feature_count, experiment_num):
    """Generates LaTeX table with proper formatting"""
    # Find best values (lower better for MAE/RMSE, higher better for R2/Spearman)
    metrics = {
        'MAE': min,
        'RMSE': min,
        'R2': max,
        'Spearman': max
    }
    
    best_values = {}
    for metric, func in metrics.items():
        valid_values = [d[metric] for d in method_data if d[metric] is not None]
        if valid_values:
            best_values[metric] = func(valid_values)
    
    # Generate LaTeX content
    latex_content = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Método} & \textbf{MAE} & \textbf{RMSE} & \textbf{R²} & \textbf{Spearman} \\",
        r"\midrule"
    ]
    
    for method in method_data:
        row = [method['Method']]
        for metric in ['MAE', 'RMSE', 'R2', 'Spearman']:
            value = method[metric]
            if value is None:
                row.append("N/A")
            else:
                # Format with 3 decimal places
                value_str = f"{value:.3f}"
                # Bold if it's the best value
                if metric in best_values and value == best_values[metric]:
                    value_str = r"\textbf{" + value_str + "}"
                row.append(value_str)
        
        latex_content.append(" & ".join(row) + r" \\")
    
    latex_content.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Resultados de test para {feature_count} features, Experimento {experiment_num}}}",
        f"\\label{{tab:test_results_{feature_count}feat_exp{experiment_num}}}",
        r"\end{table}"
    ])
    
    return "\n".join(latex_content)

def process_experiments():
    """Process all experiments for both 40 and 80 features"""
    for feature_count in [40, 80]:
        for exp_num in range(1, 4):
            # Define method files based on feature count
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
            
            method_data = []
            for method, filepath in method_files.items():
                if os.path.exists(filepath):
                    results = extract_test_results(filepath)
                    if any(results.values()):  # Only add if we got some results
                        method_data.append({
                            'Method': method,
                            'Features': '3' if 'tsne' in filepath.lower() else str(feature_count),
                            **results
                        })
                    else:
                        print(f"Skipping {method} - no valid results in {filepath}")
                else:
                    print(f"File not found: {filepath}")
            
            if method_data:
                latex_table = generate_test_latex_table(method_data, feature_count, exp_num)
                
                # Save to file
                output_dir = f"latex_test_tables/{feature_count}features"
                Path(output_dir).mkdir(exist_ok=True)
                output_file = f"{output_dir}/test_results_exp_{exp_num}.tex"
                
                with open(output_file, 'w') as f:
                    f.write(latex_table)
                print(f"Saved results to {output_file}")
            else:
                print(f"No valid data for {feature_count} features experiment {exp_num}")

if __name__ == "__main__":
    process_experiments()
    print("\nProcessing complete. LaTeX tables saved in latex_test_tables/")