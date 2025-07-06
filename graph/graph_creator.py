import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Create directories if they don't exist
Path("graph").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path("latex_tables").mkdir(exist_ok=True)

def extract_training_data(filepath):
    """Extract epoch numbers, loss values, and time from log file"""
    epochs = []
    losses = []
    times = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("--- training epoch"):
                # Extract epoch number
                epoch_match = re.search(r"epoch (\d+)", line)
                if epoch_match:
                    epochs.append(int(epoch_match.group(1)))
                
                # Extract loss value
                loss_match = re.search(r"loss (-?\d+\.\d+)", line)
                if loss_match:
                    losses.append(float(loss_match.group(1)))
                
                # Extract time elapsed
                time_match = re.search(r"time elapsed\(min\) (\d+\.\d+)", line)
                if time_match:
                    times.append(float(time_match.group(1)))
    
    return epochs, losses, times

def generate_latex_table(method_data, feature_count, experiment_num):
    """Generate LaTeX table from method data"""
    # Find best loss to bold in table
    best_loss = min([d['Final Loss'] for d in method_data])
    best_time = min([d['Total Time (min)'] for d in method_data])
    
    latex_content = r"""
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Método} & \textbf{Pérdida Final} & \textbf{Tiempo Total (min)} \\
\midrule
"""
    
    for method in method_data:
        loss = method['Final Loss']
        time = method['Total Time (min)']
        
        # Formatting with bold for best values
        loss_str = f"{loss:.3f}"
        time_str = f"{time:.2f}"
        
        if loss == best_loss:
            loss_str = r"\textbf{" + loss_str + "}"
        if time == best_time:
            time_str = r"\textbf{" + time_str + "}"
        
        latex_content += f"{method['Method']} & {loss_str} & {time_str} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\caption{Resultados para """ + f"{feature_count} features, Experimento {experiment_num}" + r"""}
\label{tab:""" + f"results_{feature_count}feat_exp{experiment_num}" + r"""}
\end{table}
"""
    
    # Save LaTeX table to file
    latex_dir = f"latex_tables/{feature_count}features"
    Path(latex_dir).mkdir(exist_ok=True)
    latex_filename = f"{latex_dir}/table_exp_{experiment_num}.tex"
    
    with open(latex_filename, 'w') as f:
        f.write(latex_content)
    
    print(f"Saved LaTeX table to {latex_filename}")
    return latex_content

def generate_comparison_graph(experiment_num, feature_count):
    """Generate comparison graph for one experiment"""
    # Define the methods and their corresponding log files
    if feature_count == 80:
        methods = {
            'PCA': f'logs/finetune_pca_80features_800epoch_{experiment_num}.log',
            'ISOMAP': f'logs/finetune_isomap_80features_800epoch_{experiment_num}.log',
            't-SNE': f'logs/finetune_tsne_3features_800epoch_{experiment_num}.log',
            'UMAP': f'logs/finetune_umap_80features_800epoch_{experiment_num}.log'
        }
        title_suffix = " (PCA/ISOMAP/UMAP: 80 Features, t-SNE: 3 Features)"
    else:  # 40 features
        methods = {
            'PCA': f'logs/finetune_pca_40features_{experiment_num}.log',
            'ISOMAP': f'logs/finetune_isomap_40features_{experiment_num}.log',
            't-SNE': f'logs/finetune_tsne_3features_{experiment_num}.log',
            'UMAP': f'logs/finetune_umap_40features_{experiment_num}.log'
        }
        title_suffix = " (PCA/ISOMAP/UMAP: 40 Features, t-SNE: 3 Features)"
    
    plt.figure(figsize=(12, 6))
    
    # Store data for the table
    table_data = []
    
    for method, logfile in methods.items():
        if not os.path.exists(logfile):
            print(f"Warning: File {logfile} not found. Skipping.")
            continue
            
        epochs, losses, times = extract_training_data(logfile)
        
        if not epochs or not losses or not times:
            print(f"Warning: No valid data found in {logfile}. Skipping.")
            continue
            
        # Plot the data
        plt.plot(epochs, losses, label=method)
        
        # Calculate total time (sum of all time elapsed entries)
        total_time = sum(times)
        
        # Add to table data
        table_data.append({
            'Method': method,
            'Features': '3' if 'tsne' in logfile.lower() else str(feature_count),
            'Final Loss': losses[-1],
            'Total Time (min)': round(total_time, 2)
        })
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison - Experiment {experiment_num}{title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Create appropriate directory structure
    feature_dir = f'graph/{feature_count}features'
    Path(feature_dir).mkdir(exist_ok=True)
    
    # Save the plot
    plot_filename = f'{feature_dir}/loss_comparison_exp_{experiment_num}.png'
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Saved plot to {plot_filename}")
    
    # Create and display summary table
    if table_data:
        df = pd.DataFrame(table_data)
        print(f"\nSummary for {feature_count}-feature Experiment {experiment_num}:")
        print(df.to_string(index=False))
        
        # Save table to CSV
        csv_filename = f'{feature_dir}/summary_exp_{experiment_num}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Saved summary to {csv_filename}")
        
        # Generate LaTeX table
        latex_table = generate_latex_table(table_data, feature_count, experiment_num)
        print("\nGenerated LaTeX table:")
        print(latex_table)

# Generate graphs and tables for all experiments
for feature_count in [40, 80]:
    for exp_num in range(1, 4):
        generate_comparison_graph(exp_num, feature_count)

print("\nAll outputs generated:")
print("- Graphs in /graph/[40|80]features/")
print("- CSV summaries in /graph/[40|80]features/")
print("- LaTeX tables in /latex_tables/[40|80]features/")