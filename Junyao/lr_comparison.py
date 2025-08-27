#!/usr/bin/env python3
"""
Learning Rate Strategy Comparison Tool
Runs multiple LR strategies and compares their performance with charts and tables
"""

import os
import json
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LRComparisonRunner:
    """Runs multiple learning rate strategies and compares results"""
    
    def __init__(self, base_config_file: str = "training_config.py"):
        self.base_config_file = base_config_file
        self.results_dir = "lr_comparison_results"
        self.results = {}
        self.comparison_data = []
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define strategies to test
        self.strategies = {
            'reduce_lr_on_plateau': {
                'name': 'Reduce LR on Plateau',
                'description': 'Reduces LR when loss plateaus',
                'color': '#1f77b4'
            },
            'cosine': {
                'name': 'Cosine Annealing',
                'description': 'Smooth cosine decay schedule',
                'color': '#ff7f0e'
            },
            'step': {
                'name': 'Step Decay',
                'description': 'Step reduction every N epochs',
                'color': '#2ca02c'
            },
            'exponential': {
                'name': 'Exponential Decay',
                'description': 'Continuous exponential decay',
                'color': '#d62728'
            },
            'one_cycle': {
                'name': 'One-Cycle Policy',
                'description': 'Fast training with momentum cycling',
                'color': '#9467bd'
            },
            'none': {
                'name': 'No Scheduling',
                'description': 'Constant learning rate',
                'color': '#8c564b'
            }
        }
    
    def backup_config(self) -> str:
        """Backup the current configuration file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{self.base_config_file}.backup_{timestamp}"
        
        if os.path.exists(self.base_config_file):
            with open(self.base_config_file, 'r') as f:
                content = f.read()
            with open(backup_file, 'w') as f:
                f.write(content)
            print(f"✅ Backed up config to: {backup_file}")
            return backup_file
        return None
    
    def restore_config(self, backup_file: str):
        """Restore configuration from backup"""
        if backup_file and os.path.exists(backup_file):
            with open(backup_file, 'r') as f:
                content = f.read()
            with open(self.base_config_file, 'w') as f:
                f.write(content)
            print(f"✅ Restored config from: {backup_file}")
    
    def configure_strategy(self, strategy_name: str):
        """Configure a specific learning rate strategy"""
        try:
            # Use the configure_lr.py script to set the strategy
            result = subprocess.run([
                'python3', 'configure_lr.py', 'strategy', strategy_name
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ Configured strategy: {strategy_name}")
                return True
            else:
                print(f"❌ Failed to configure {strategy_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout configuring {strategy_name}")
            return False
        except Exception as e:
            print(f"❌ Error configuring {strategy_name}: {e}")
            return False
    
    def run_training(self, strategy_name: str, max_epochs: int = 10) -> Dict[str, Any]:
        """Run training for a specific strategy"""
        print(f"\n🚀 Running training for: {strategy_name}")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Run the training script
            result = subprocess.run([
                'python3', 'train_vae_1d.py'
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            training_time = time.time() - start_time
            
            # Parse results
            strategy_result = {
                'strategy': strategy_name,
                'name': self.strategies[strategy_name]['name'],
                'training_time': training_time,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'final_loss': None,
                'best_loss': None,
                'loss_history': [],
                'lr_history': [],
                'convergence_epoch': None
            }
            
            if result.returncode == 0:
                # Try to extract loss data
                strategy_result.update(self.extract_training_data(strategy_name))
                print(f"✅ Training completed for {strategy_name}")
            else:
                print(f"❌ Training failed for {strategy_name}")
                print(f"Error: {result.stderr}")
            
            return strategy_result
            
        except subprocess.TimeoutExpired:
            print(f"❌ Training timeout for {strategy_name}")
            return {
                'strategy': strategy_name,
                'name': self.strategies[strategy_name]['name'],
                'training_time': time.time() - start_time,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"❌ Training error for {strategy_name}: {e}")
            return {
                'strategy': strategy_name,
                'name': self.strategies[strategy_name]['name'],
                'training_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def extract_training_data(self, strategy_name: str) -> Dict[str, Any]:
        """Extract training data from saved files"""
        data = {}
        
        # Try to load loss data
        loss_file = 'vae_1d_training_loss_data.json'
        if os.path.exists(loss_file):
            try:
                with open(loss_file, 'r') as f:
                    loss_data = json.load(f)
                data['final_loss'] = loss_data.get('final_loss')
                data['best_loss'] = loss_data.get('best_loss')
                data['loss_history'] = loss_data.get('losses', [])
                data['convergence_epoch'] = loss_data.get('best_epoch')
            except Exception as e:
                print(f"⚠️  Could not load loss data: {e}")
        
        # Try to load LR history
        lr_file = 'vae_1d_lr_history.json'
        if os.path.exists(lr_file):
            try:
                with open(lr_file, 'r') as f:
                    lr_data = json.load(f)
                data['lr_history'] = lr_data.get('learning_rates', [])
            except Exception as e:
                print(f"⚠️  Could not load LR history: {e}")
        
        return data
    
    def run_comparison(self, strategies: List[str] = None, max_epochs: int = 10) -> Dict[str, Any]:
        """Run comparison of multiple learning rate strategies"""
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        print(f"🔬 Starting Learning Rate Strategy Comparison")
        print(f"Strategies to test: {strategies}")
        print(f"Max epochs per strategy: {max_epochs}")
        print("=" * 60)
        
        # Backup current configuration
        backup_file = self.backup_config()
        
        try:
            # Run each strategy
            for strategy in strategies:
                if strategy not in self.strategies:
                    print(f"⚠️  Skipping unknown strategy: {strategy}")
                    continue
                
                # Configure the strategy
                if not self.configure_strategy(strategy):
                    print(f"⚠️  Skipping {strategy} due to configuration failure")
                    continue
                
                # Run training
                result = self.run_training(strategy, max_epochs)
                self.results[strategy] = result
                
                # Save individual results
                result_file = os.path.join(self.results_dir, f"{strategy}_results.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"📊 Results saved to: {result_file}")
                
                # Clean up training files
                self.cleanup_training_files()
                
        finally:
            # Restore original configuration
            self.restore_config(backup_file)
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    def cleanup_training_files(self):
        """Clean up training output files"""
        files_to_clean = [
            'vae_1d_training_loss_data.json',
            'vae_1d_lr_history.json',
            'vae_1d_attention_best.pth',
            'vae_1d_attention_final.pth',
            'vae_1d_training_results.png',
            'vae_1d_training_loss.png',
            'vae_1d_lr_history.png',
            'vae_1d_reconstruction_comparison.png'
        ]
        
        for file in files_to_clean:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"⚠️  Could not remove {file}: {e}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report with charts and tables"""
        print(f"\n📊 Generating Comparison Report")
        print("=" * 50)
        
        # Create comparison data
        self.create_comparison_data()
        
        # Generate charts
        self.plot_comparison_charts()
        
        # Generate tables
        self.create_comparison_tables()
        
        # Save comprehensive report
        self.save_comprehensive_report()
        
        print(f"✅ Comparison report saved to: {self.results_dir}/")
    
    def create_comparison_data(self):
        """Create structured data for comparison"""
        self.comparison_data = []
        
        for strategy, result in self.results.items():
            if not result.get('success', False):
                continue
            
            data = {
                'Strategy': result['name'],
                'Strategy_Key': strategy,
                'Final_Loss': result.get('final_loss', float('inf')),
                'Best_Loss': result.get('best_loss', float('inf')),
                'Training_Time': result.get('training_time', 0),
                'Convergence_Epoch': result.get('convergence_epoch', 0),
                'Loss_History': result.get('loss_history', []),
                'LR_History': result.get('lr_history', []),
                'Success': result.get('success', False)
            }
            
            # Calculate additional metrics
            if data['Loss_History']:
                data['Loss_Reduction'] = data['Loss_History'][0] - data['Loss_History'][-1]
                data['Convergence_Speed'] = data['Convergence_Epoch'] if data['Convergence_Epoch'] else len(data['Loss_History'])
                data['Stability'] = np.std(data['Loss_History'][-5:]) if len(data['Loss_History']) >= 5 else np.std(data['Loss_History'])
            else:
                data['Loss_Reduction'] = 0
                data['Convergence_Speed'] = 0
                data['Stability'] = float('inf')
            
            self.comparison_data.append(data)
    
    def plot_comparison_charts(self):
        """Generate comprehensive comparison charts"""
        if not self.comparison_data:
            print("⚠️  No data to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Loss comparison over epochs
        plt.subplot(3, 3, 1)
        for data in self.comparison_data:
            if data['Loss_History']:
                epochs = range(1, len(data['Loss_History']) + 1)
                plt.plot(epochs, data['Loss_History'], 
                        label=data['Strategy'], 
                        color=self.strategies[data['Strategy_Key']]['color'],
                        linewidth=2)
        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. Learning rate comparison
        plt.subplot(3, 3, 2)
        for data in self.comparison_data:
            if data['LR_History']:
                epochs = range(1, len(data['LR_History']) + 1)
                plt.plot(epochs, data['LR_History'], 
                        label=data['Strategy'],
                        color=self.strategies[data['Strategy_Key']]['color'],
                        linewidth=2)
        plt.title('Learning Rate Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. Final loss comparison
        plt.subplot(3, 3, 3)
        strategies = [data['Strategy'] for data in self.comparison_data]
        final_losses = [data['Final_Loss'] for data in self.comparison_data]
        colors = [self.strategies[data['Strategy_Key']]['color'] for data in self.comparison_data]
        
        bars = plt.bar(strategies, final_losses, color=colors, alpha=0.7)
        plt.title('Final Loss Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Final Loss')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, loss in zip(bars, final_losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_losses)*0.01,
                    f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Training time comparison
        plt.subplot(3, 3, 4)
        training_times = [data['Training_Time'] for data in self.comparison_data]
        bars = plt.bar(strategies, training_times, color=colors, alpha=0.7)
        plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 5. Convergence speed comparison
        plt.subplot(3, 3, 5)
        convergence_speeds = [data['Convergence_Speed'] for data in self.comparison_data]
        bars = plt.bar(strategies, convergence_speeds, color=colors, alpha=0.7)
        plt.title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Convergence Epoch')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, speed in zip(bars, convergence_speeds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(convergence_speeds)*0.01,
                    f'{speed}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Loss reduction comparison
        plt.subplot(3, 3, 6)
        loss_reductions = [data['Loss_Reduction'] for data in self.comparison_data]
        bars = plt.bar(strategies, loss_reductions, color=colors, alpha=0.7)
        plt.title('Loss Reduction Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Loss Reduction')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, reduction in zip(bars, loss_reductions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(loss_reductions)*0.01,
                    f'{reduction:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Stability comparison (lower is better)
        plt.subplot(3, 3, 7)
        stabilities = [data['Stability'] for data in self.comparison_data]
        bars = plt.bar(strategies, stabilities, color=colors, alpha=0.7)
        plt.title('Training Stability Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Loss Std Dev (Last 5 Epochs)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, stability in zip(bars, stabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stabilities)*0.01,
                    f'{stability:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Performance radar chart
        plt.subplot(3, 3, 8)
        self.plot_radar_chart()
        
        # 9. Summary statistics
        plt.subplot(3, 3, 9)
        self.plot_summary_stats()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'lr_comparison_comprehensive.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create individual detailed plots
        self.create_detailed_plots()
    
    def plot_radar_chart(self):
        """Create a radar chart comparing multiple metrics"""
        if len(self.comparison_data) < 2:
            plt.text(0.5, 0.5, 'Need at least 2 strategies\nfor radar chart', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Performance Radar Chart', fontsize=14, fontweight='bold')
            return
        
        # Normalize metrics for radar chart
        metrics = ['Final_Loss', 'Training_Time', 'Convergence_Speed', 'Loss_Reduction', 'Stability']
        metric_names = ['Final Loss\n(Lower Better)', 'Training Time\n(Lower Better)', 
                       'Convergence\n(Lower Better)', 'Loss Reduction\n(Higher Better)', 
                       'Stability\n(Lower Better)']
        
        # Normalize data (0-1 scale, where 1 is best)
        normalized_data = []
        for data in self.comparison_data:
            normalized = []
            for metric in metrics:
                if metric == 'Loss_Reduction':
                    # Higher is better
                    normalized.append(data[metric] / max(d['Loss_Reduction'] for d in self.comparison_data))
                else:
                    # Lower is better
                    normalized.append(1 - (data[metric] / max(d[metric] for d in self.comparison_data)))
            normalized_data.append(normalized)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax = plt.subplot(111, projection='polar')
        for i, data in enumerate(self.comparison_data):
            values = normalized_data[i] + normalized_data[i][:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=data['Strategy'], 
                   color=self.strategies[data['Strategy_Key']]['color'])
            ax.fill(angles, values, alpha=0.25, 
                   color=self.strategies[data['Strategy_Key']]['color'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def plot_summary_stats(self):
        """Plot summary statistics"""
        plt.axis('off')
        
        # Create summary text
        summary_text = "Comparison Summary:\n\n"
        
        if self.comparison_data:
            # Find best performers
            best_final_loss = min(self.comparison_data, key=lambda x: x['Final_Loss'])
            fastest_training = min(self.comparison_data, key=lambda x: x['Training_Time'])
            fastest_convergence = min(self.comparison_data, key=lambda x: x['Convergence_Speed'])
            most_stable = min(self.comparison_data, key=lambda x: x['Stability'])
            
            summary_text += f"🏆 Best Final Loss: {best_final_loss['Strategy']}\n"
            summary_text += f"⚡ Fastest Training: {fastest_training['Strategy']}\n"
            summary_text += f"🎯 Fastest Convergence: {fastest_convergence['Strategy']}\n"
            summary_text += f"📊 Most Stable: {most_stable['Strategy']}\n\n"
            
            summary_text += f"Total Strategies Tested: {len(self.comparison_data)}\n"
            summary_text += f"Successful Runs: {sum(1 for d in self.comparison_data if d['Success'])}\n"
            summary_text += f"Average Training Time: {np.mean([d['Training_Time'] for d in self.comparison_data]):.1f}s"
        
        plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    def create_detailed_plots(self):
        """Create individual detailed plots for each metric"""
        if not self.comparison_data:
            return
        
        # 1. Loss curves comparison (detailed)
        plt.figure(figsize=(12, 8))
        for data in self.comparison_data:
            if data['Loss_History']:
                epochs = range(1, len(data['Loss_History']) + 1)
                plt.plot(epochs, data['Loss_History'], 
                        label=f"{data['Strategy']} (Final: {data['Final_Loss']:.4f})", 
                        color=self.strategies[data['Strategy_Key']]['color'],
                        linewidth=2, marker='o', markersize=4)
        
        plt.title('Detailed Training Loss Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'detailed_loss_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Learning rate schedules comparison
        plt.figure(figsize=(12, 8))
        for data in self.comparison_data:
            if data['LR_History']:
                epochs = range(1, len(data['LR_History']) + 1)
                plt.plot(epochs, data['LR_History'], 
                        label=data['Strategy'], 
                        color=self.strategies[data['Strategy_Key']]['color'],
                        linewidth=2, marker='s', markersize=4)
        
        plt.title('Learning Rate Schedule Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'lr_schedule_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_tables(self):
        """Create comparison tables"""
        if not self.comparison_data:
            print("⚠️  No data for tables")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.comparison_data)
        
        # Select relevant columns for display
        display_columns = ['Strategy', 'Final_Loss', 'Best_Loss', 'Training_Time', 
                          'Convergence_Speed', 'Loss_Reduction', 'Stability']
        
        # Create formatted table
        table_df = df[display_columns].copy()
        table_df.columns = ['Strategy', 'Final Loss', 'Best Loss', 'Training Time (s)', 
                           'Convergence Epoch', 'Loss Reduction', 'Stability']
        
        # Format numeric columns
        table_df['Final Loss'] = table_df['Final Loss'].apply(lambda x: f"{x:.4f}")
        table_df['Best Loss'] = table_df['Best Loss'].apply(lambda x: f"{x:.4f}")
        table_df['Training Time (s)'] = table_df['Training Time (s)'].apply(lambda x: f"{x:.1f}")
        table_df['Loss Reduction'] = table_df['Loss Reduction'].apply(lambda x: f"{x:.4f}")
        table_df['Stability'] = table_df['Stability'].apply(lambda x: f"{x:.4f}")
        
        # Save table
        table_file = os.path.join(self.results_dir, 'comparison_table.csv')
        table_df.to_csv(table_file, index=False)
        
        # Create HTML table with styling
        html_table = table_df.to_html(index=False, classes='table table-striped table-hover')
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Learning Rate Strategy Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; font-weight: bold; }}
                .table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .table tr:hover {{ background-color: #f5f5f5; }}
                h1 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Learning Rate Strategy Comparison Results</h1>
            {html_table}
        </body>
        </html>
        """
        
        html_file = os.path.join(self.results_dir, 'comparison_table.html')
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Comparison tables saved:")
        print(f"   - {table_file}")
        print(f"   - {html_file}")
        
        # Print table to console
        print(f"\n📊 Comparison Results Table:")
        print("=" * 80)
        print(table_df.to_string(index=False))
        print("=" * 80)
    
    def save_comprehensive_report(self):
        """Save comprehensive comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'strategies_tested': list(self.results.keys()),
            'results': self.results,
            'comparison_data': self.comparison_data,
            'summary': self.generate_summary()
        }
        
        report_file = os.path.join(self.results_dir, 'comprehensive_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved to: {report_file}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.comparison_data:
            return {}
        
        successful_runs = [d for d in self.comparison_data if d['Success']]
        
        if not successful_runs:
            return {'error': 'No successful runs'}
        
        summary = {
            'total_strategies': len(self.comparison_data),
            'successful_runs': len(successful_runs),
            'best_final_loss': {
                'strategy': min(successful_runs, key=lambda x: x['Final_Loss'])['Strategy'],
                'value': min(successful_runs, key=lambda x: x['Final_Loss'])['Final_Loss']
            },
            'fastest_training': {
                'strategy': min(successful_runs, key=lambda x: x['Training_Time'])['Strategy'],
                'value': min(successful_runs, key=lambda x: x['Training_Time'])['Training_Time']
            },
            'fastest_convergence': {
                'strategy': min(successful_runs, key=lambda x: x['Convergence_Speed'])['Strategy'],
                'value': min(successful_runs, key=lambda x: x['Convergence_Speed'])['Convergence_Speed']
            },
            'most_stable': {
                'strategy': min(successful_runs, key=lambda x: x['Stability'])['Strategy'],
                'value': min(successful_runs, key=lambda x: x['Stability'])['Stability']
            },
            'average_metrics': {
                'final_loss': np.mean([d['Final_Loss'] for d in successful_runs]),
                'training_time': np.mean([d['Training_Time'] for d in successful_runs]),
                'convergence_speed': np.mean([d['Convergence_Speed'] for d in successful_runs]),
                'loss_reduction': np.mean([d['Loss_Reduction'] for d in successful_runs]),
                'stability': np.mean([d['Stability'] for d in successful_runs])
            }
        }
        
        return summary

def main():
    """Main function for running LR comparison"""
    print("🔬 Learning Rate Strategy Comparison Tool")
    print("=" * 50)
    
    # Create comparison runner
    runner = LRComparisonRunner()
    
    # Define strategies to test
    strategies_to_test = [
        'reduce_lr_on_plateau',
        'cosine', 
        'step',
        'exponential',
        'one_cycle',
        'none'
    ]
    
    print(f"Strategies to test: {strategies_to_test}")
    print(f"Results will be saved to: {runner.results_dir}/")
    
    # Run comparison
    results = runner.run_comparison(strategies=strategies_to_test, max_epochs=10)
    
    print(f"\n✅ Comparison completed!")
    print(f"Check the results in: {runner.results_dir}/")

if __name__ == "__main__":
    main()
