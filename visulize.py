import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set tracking URI
mlflow.set_tracking_uri("mlruns")

# Get MLflow client
client = mlflow.tracking.MlflowClient()

# Get all experiments
experiments = client.search_experiments()

print("=== MLflow Experiments Summary ===\n")

for exp in experiments:
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
    
    # Get all runs for this experiment
    runs = client.search_runs(exp.experiment_id)
    
    if not runs:
        print("  No runs found\n")
        continue
    
    # Collect data for visualization
    data = []
    for run in runs:
        run_data = {
            'run_name': run.info.run_name,
            'run_id': run.info.run_id[:8],
            'status': run.info.status,
        }
        
        # Add all metrics
        for metric_key, metric_value in run.data.metrics.items():
            run_data[metric_key] = metric_value
        
        # Add all parameters
        for param_key, param_value in run.data.params.items():
            run_data[f'param_{param_key}'] = param_value
        
        data.append(run_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print summary table
    print("\n--- Runs Summary ---")
    print(df[['run_name', 'accuracy', 'f1_score', 'precision', 'recall']].to_string(index=False))
    print()
    
    # Create visualizations
    if len(df) > 0:
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'MLflow Experiment: {exp.name}', fontsize=16)
            
            for idx, metric in enumerate(available_metrics):
                ax = axes[idx // 2, idx % 2]
                df_sorted = df.sort_values(metric, ascending=False)
                
                bars = ax.barh(df_sorted['run_name'], df_sorted[metric])
                ax.set_xlabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} Comparison')
                ax.set_xlim([0, 1])
                
                # Color the best performer
                bars[0].set_color('green')
                
                # Add value labels
                for i, (name, value) in enumerate(zip(df_sorted['run_name'], df_sorted[metric])):
                    ax.text(value + 0.01, i, f'{value:.4f}', va='center')
            
            plt.tight_layout()
            plt.savefig('mlflow_results_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Saved visualization: mlflow_results_comparison.png\n")
            
            # Create detailed metrics table visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = df[['run_name'] + available_metrics].round(4)
            table = ax.table(cellText=table_data.values, 
                           colLabels=table_data.columns,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Highlight best scores
            for i in range(len(available_metrics)):
                best_idx = df[available_metrics[i]].idxmax()
                table[(best_idx + 1, i + 1)].set_facecolor('#90EE90')
            
            plt.title('MLflow Experiment Results', fontsize=14, pad=20)
            plt.savefig('mlflow_results_table.png', dpi=300, bbox_inches='tight')
            print(f"Saved table: mlflow_results_table.png\n")

print("\n=== MLflow artifacts location ===")
print(f"All experiment data stored in: mlruns/")
print("\nYou can share these visualizations to demonstrate MLflow tracking!")