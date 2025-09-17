#!/usr/bin/env python3
"""
Parallel Learning Rate Strategy Comparison Tool
Runs multiple LR strategies simultaneously using multiple threads
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
import threading
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
warnings.filterwarnings('ignore')

class ParallelLRComparisonRunner:
    """Runs multiple learning rate strategies in parallel"""
    
    def __init__(self, base_config_file: str = "Junyao/training_config.py", max_workers: int = None):
        self.base_config_file = base_config_file
        self.results_dir = "parallel_lr_comparison_results"
        self.results = {}
        self.comparison_data = []
        self.results_lock = threading.Lock()
        
        # Determine number of workers
        if max_workers is None:
            # Use CPU count, but leave some cores free for system
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.max_workers = max_workers
        
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
        
        print(f"🖥️  System Info:")
        print(f"   CPU Cores: {multiprocessing.cpu_count()}")
        print(f"   Available Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"   Parallel Workers: {self.max_workers}")
    
    def create_strategy_config(self, strategy_name: str) -> str:
        """Create a temporary config file for a specific strategy"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = f"Junyao/training_config_{strategy_name}_{timestamp}.py"
        
        # Read base config
        with open(self.base_config_file, 'r') as f:
            content = f.read()
        
        # Update strategy
        import re
        content = re.sub(
            r"LR_SCHEDULER_TYPE = '[^']*'",
            f"LR_SCHEDULER_TYPE = '{strategy_name}'",
            content
        )
        
        # Update scheduler parameters based on strategy
        strategy_params = self.get_strategy_params(strategy_name)
        if strategy_params:
            # Find and replace the scheduler parameters section
            start_marker = "LR_SCHEDULER_PARAMS = {"
            start_idx = content.find(start_marker)
            if start_idx != -1:
                # Find the matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(content[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                # Create new parameters string
                params_str = "LR_SCHEDULER_PARAMS = {\n"
                for key, value in strategy_params.items():
                    if isinstance(value, str):
                        params_str += f"        '{key}': '{value}',\n"
                    elif isinstance(value, bool):
                        params_str += f"        '{key}': {str(value)},\n"
                    elif value is None:
                        params_str += f"        '{key}': None,\n"
                    else:
                        params_str += f"        '{key}': {value},\n"
                params_str += "    }"
                
                # Replace the parameters section
                content = content[:start_idx] + params_str + content[end_idx+1:]
        
        # Write temporary config
        with open(config_file, 'w') as f:
            f.write(content)
        
        return config_file
    
    def get_strategy_params(self, strategy_name: str) -> Dict:
        """Get parameters for a specific strategy"""
        params = {
            'reduce_lr_on_plateau': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6,
                'verbose': True,
                'threshold': 1e-4,
                'threshold_mode': 'rel'
            },
            'cosine': {
                'T_max': 50,
                'eta_min': 1e-6,
                'last_epoch': -1
            },
            'step': {
                'step_size': 10,
                'gamma': 0.5,
                'last_epoch': -1
            },
            'exponential': {
                'gamma': 0.95,
                'last_epoch': -1
            },
            'one_cycle': {
                'max_lr': 1e-2,
                'total_steps': None,
                'epochs': None,
                'steps_per_epoch': None,
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'cycle_momentum': True,
                'base_momentum': 0.85,
                'max_momentum': 0.95,
                'div_factor': 25.0,
                'final_div_factor': 1e4
            },
            'none': {}
        }
        
        return params.get(strategy_name, {})
    
    def run_strategy_parallel(self, strategy_name: str, max_epochs: int = 10) -> Dict[str, Any]:
        """Run a single strategy in parallel (thread-safe)"""
        print(f"🚀 Starting parallel training for: {strategy_name}")
        
        start_time = time.time()
        
        # Create temporary config file for this strategy
        temp_config = self.create_strategy_config(strategy_name)
        
        try:
            # Create temporary training script that uses the temp config
            temp_train_script = f"train_vae_1d_{strategy_name}.py"
            self.create_temp_training_script(temp_train_script, temp_config)
            
            # Run training with the temporary config from the correct directory
            current_dir = os.path.abspath(os.path.dirname(__file__))
            result = subprocess.run([
                'python3', temp_train_script
            ], capture_output=True, text=True, timeout=3600, cwd=current_dir)  # 1 hour timeout
            
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
                # Extract training data
                strategy_result.update(self.extract_training_data(strategy_name))
                print(f"✅ Parallel training completed for {strategy_name}")
            else:
                print(f"❌ Parallel training failed for {strategy_name}")
                print(f"Error: {result.stderr}")
            
            # Store result thread-safely
            with self.results_lock:
                self.results[strategy_name] = strategy_result
            
            return strategy_result
            
        except subprocess.TimeoutExpired:
            print(f"❌ Parallel training timeout for {strategy_name}")
            return {
                'strategy': strategy_name,
                'name': self.strategies[strategy_name]['name'],
                'training_time': time.time() - start_time,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"❌ Parallel training error for {strategy_name}: {e}")
            return {
                'strategy': strategy_name,
                'name': self.strategies[strategy_name]['name'],
                'training_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up temporary files
            self.cleanup_temp_files(temp_config, temp_train_script)
    
    def create_temp_training_script(self, script_name: str, config_file: str):
        """Create a temporary training script that uses a specific config file"""
        # Get the absolute path to the current directory
        current_dir = os.path.abspath(os.path.dirname(__file__))
        
        script_content = f'''#!/usr/bin/env python3
"""
Temporary training script for parallel execution
Uses specific config file: {config_file}
"""

import sys
import os

# Add the Junyao directory to Python path
current_dir = "{current_dir}"
sys.path.insert(0, current_dir)

# Change to the correct directory
os.chdir(current_dir)

# Import training modules
from train_vae_1d import train_vae, plot_training_results

# Override config import to use specific config file
import importlib.util
spec = importlib.util.spec_from_file_location("temp_config", "{config_file}")
temp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temp_config)

# Override global variables with temp config values
globals().update({{k: v for k, v in temp_config.__dict__.items() 
                 if not k.startswith('_') and k.isupper()}})

if __name__ == "__main__":
    # Run training with overridden config
    model, losses = train_vae()
    print(f"Training completed for config: {{config_file}}")
'''
        
        with open('Junyao/'+script_name, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_name, 0o755)
    
    def cleanup_temp_files(self, config_file: str, script_file: str):
        """Clean up temporary files"""
        for file in [config_file, script_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"⚠️  Could not remove {file}: {e}")
    
    def extract_training_data(self, strategy_name: str) -> Dict[str, Any]:
        """Extract training data from saved files"""
        data = {}
        
        # Try to load loss data (with strategy-specific naming)
        loss_file = f'vae_1d_training_loss_data_{strategy_name}.json'
        if os.path.exists(loss_file):
            try:
                with open(loss_file, 'r') as f:
                    loss_data = json.load(f)
                data['final_loss'] = loss_data.get('final_loss')
                data['best_loss'] = loss_data.get('best_loss')
                data['loss_history'] = loss_data.get('losses', [])
                data['convergence_epoch'] = loss_data.get('best_epoch')
            except Exception as e:
                print(f"⚠️  Could not load loss data for {strategy_name}: {e}")
        
        # Try to load LR history
        lr_file = f'vae_1d_lr_history_{strategy_name}.json'
        if os.path.exists(lr_file):
            try:
                with open(lr_file, 'r') as f:
                    lr_data = json.load(f)
                data['lr_history'] = lr_data.get('learning_rates', [])
            except Exception as e:
                print(f"⚠️  Could not load LR history for {strategy_name}: {e}")
        
        return data
    
    def run_parallel_comparison(self, strategies: List[str] = None, max_epochs: int = 10) -> Dict[str, Any]:
        """Run comparison of multiple learning rate strategies in parallel"""
        if strategies is None:
            strategies = list(self.strategies.keys())
        
        print(f"🔬 Starting Parallel Learning Rate Strategy Comparison")
        print(f"Strategies to test: {strategies}")
        print(f"Max epochs per strategy: {max_epochs}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"Estimated speedup: {len(strategies)}x (vs sequential)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run strategies in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_strategy = {
                executor.submit(self.run_strategy_parallel, strategy, max_epochs): strategy
                for strategy in strategies
            }
            
            # Process completed tasks
            completed_count = 0
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    completed_count += 1
                    print(f"📊 Completed {completed_count}/{len(strategies)}: {strategy}")
                    
                    # Save individual results
                    result_file = os.path.join(self.results_dir, f"{strategy}_results.json")
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    
                except Exception as e:
                    print(f"❌ Exception for {strategy}: {e}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Parallel comparison completed in {total_time:.1f} seconds")
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report with charts and tables"""
        print(f"\n📊 Generating Parallel Comparison Report")
        print("=" * 50)
        
        # Create comparison data
        self.create_comparison_data()
        
        # Generate charts
        self.plot_comparison_charts()
        
        # Generate tables
        self.create_comparison_tables()
        
        # Save comprehensive report
        self.save_comprehensive_report()
        
        print(f"✅ Parallel comparison report saved to: {self.results_dir}/")
    
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
        plt.title('Parallel Training Loss Comparison', fontsize=14, fontweight='bold')
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
        plt.title('Final Loss Comparison (Parallel)', fontsize=14, fontweight='bold')
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
        plt.title('Training Time Comparison (Parallel)', fontsize=14, fontweight='bold')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 5. Parallel efficiency
        plt.subplot(3, 3, 5)
        total_sequential_time = sum(training_times)
        parallel_time = max(training_times)  # Longest training time
        efficiency = (total_sequential_time / parallel_time) / self.max_workers * 100
        
        plt.bar(['Sequential', 'Parallel'], [total_sequential_time, parallel_time], 
               color=['red', 'green'], alpha=0.7)
        plt.title(f'Parallel Efficiency: {efficiency:.1f}%', fontsize=14, fontweight='bold')
        plt.ylabel('Total Time (seconds)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        plt.text(0, total_sequential_time + max([total_sequential_time, parallel_time])*0.01,
                f'{total_sequential_time:.1f}s', ha='center', va='bottom', fontweight='bold')
        plt.text(1, parallel_time + max([total_sequential_time, parallel_time])*0.01,
                f'{parallel_time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 6. Speedup comparison
        plt.subplot(3, 3, 6)
        speedup = total_sequential_time / parallel_time
        plt.bar(['Speedup'], [speedup], color='blue', alpha=0.7)
        plt.title(f'Speedup: {speedup:.1f}x', fontsize=14, fontweight='bold')
        plt.ylabel('Speedup Factor')
        plt.grid(True, alpha=0.3, axis='y')
        plt.text(0, speedup + speedup*0.01, f'{speedup:.1f}x', 
                ha='center', va='bottom', fontweight='bold')
        
        # 7. Resource utilization
        plt.subplot(3, 3, 7)
        utilization = (len(self.comparison_data) / self.max_workers) * 100
        plt.pie([utilization, 100-utilization], labels=['Used', 'Idle'], 
               colors=['green', 'lightgray'], autopct='%1.1f%%')
        plt.title(f'Resource Utilization: {utilization:.1f}%', fontsize=14, fontweight='bold')
        
        # 8. Performance summary
        plt.subplot(3, 3, 8)
        self.plot_performance_summary()
        
        # 9. System info
        plt.subplot(3, 3, 9)
        self.plot_system_info()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'parallel_lr_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_summary(self):
        """Plot performance summary"""
        plt.axis('off')
        
        if self.comparison_data:
            best_final_loss = min(self.comparison_data, key=lambda x: x['Final_Loss'])
            fastest_training = min(self.comparison_data, key=lambda x: x['Training_Time'])
            
            summary_text = f"""
            Parallel Performance Summary:
            
            🏆 Best Final Loss: {best_final_loss['Strategy']}
            ⚡ Fastest Training: {fastest_training['Strategy']}
            
            📊 Parallel Metrics:
            • Workers Used: {self.max_workers}
            • Strategies Tested: {len(self.comparison_data)}
            • Total Sequential Time: {sum(d['Training_Time'] for d in self.comparison_data):.1f}s
            • Parallel Time: {max(d['Training_Time'] for d in self.comparison_data):.1f}s
            • Speedup: {sum(d['Training_Time'] for d in self.comparison_data) / max(d['Training_Time'] for d in self.comparison_data):.1f}x
            """
        else:
            summary_text = "No data available"
        
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    def plot_system_info(self):
        """Plot system information"""
        plt.axis('off')
        
        system_text = f"""
        System Information:
        
        🖥️ Hardware:
        • CPU Cores: {multiprocessing.cpu_count()}
        • Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
        • Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB
        
        ⚙️ Configuration:
        • Parallel Workers: {self.max_workers}
        • Max Workers: {multiprocessing.cpu_count()}
        • Memory Usage: {psutil.virtual_memory().percent:.1f}%
        """
        
        plt.text(0.1, 0.5, system_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
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
        table_file = os.path.join(self.results_dir, 'parallel_comparison_table.csv')
        table_df.to_csv(table_file, index=False)
        
        print(f"✅ Parallel comparison table saved: {table_file}")
        
        # Print table to console
        print(f"\n📊 Parallel Comparison Results:")
        print("=" * 80)
        print(table_df.to_string(index=False))
        print("=" * 80)
    
    def save_comprehensive_report(self):
        """Save comprehensive comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'parallel_workers': self.max_workers,
            'strategies_tested': list(self.results.keys()),
            'results': self.results,
            'comparison_data': self.comparison_data,
            'summary': self.generate_summary()
        }
        
        report_file = os.path.join(self.results_dir, 'parallel_comprehensive_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Parallel comprehensive report saved to: {report_file}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.comparison_data:
            return {}
        
        successful_runs = [d for d in self.comparison_data if d['Success']]
        
        if not successful_runs:
            return {'error': 'No successful runs'}
        
        total_sequential_time = sum(d['Training_Time'] for d in successful_runs)
        parallel_time = max(d['Training_Time'] for d in successful_runs)
        speedup = total_sequential_time / parallel_time
        efficiency = (speedup / self.max_workers) * 100
        
        summary = {
            'total_strategies': len(self.comparison_data),
            'successful_runs': len(successful_runs),
            'parallel_workers': self.max_workers,
            'total_sequential_time': total_sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'best_final_loss': {
                'strategy': min(successful_runs, key=lambda x: x['Final_Loss'])['Strategy'],
                'value': min(successful_runs, key=lambda x: x['Final_Loss'])['Final_Loss']
            },
            'fastest_training': {
                'strategy': min(successful_runs, key=lambda x: x['Training_Time'])['Strategy'],
                'value': min(successful_runs, key=lambda x: x['Training_Time'])['Training_Time']
            }
        }
        
        return summary

def main():
    """Main function for running parallel LR comparison"""
    print("🔬 Parallel Learning Rate Strategy Comparison Tool")
    print("=" * 60)
    
    # Get user input for parallel configuration
    print("System Information:")
    print(f"  CPU Cores: {multiprocessing.cpu_count()}")
    print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Ask for number of workers
    try:
        max_workers = int(input(f"\nNumber of parallel workers (default {max(1, multiprocessing.cpu_count() - 1)}): ").strip() or max(1, multiprocessing.cpu_count() - 1))
    except ValueError:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create parallel comparison runner
    runner = ParallelLRComparisonRunner(max_workers=max_workers)
    
    # Define strategies to test
    strategies_to_test = [
        'reduce_lr_on_plateau',
        'cosine', 
        'step',
        'exponential',
        'one_cycle',
        'none'
    ]
    
    print(f"\nStrategies to test: {strategies_to_test}")
    print(f"Parallel workers: {max_workers}")
    print(f"Results will be saved to: {runner.results_dir}/")
    
    # Estimate time
    estimated_sequential_time = len(strategies_to_test) * 10 * 60  # 10 epochs * 60 seconds per epoch
    estimated_parallel_time = estimated_sequential_time / max_workers
    print(f"Estimated sequential time: {estimated_sequential_time/60:.1f} minutes")
    print(f"Estimated parallel time: {estimated_parallel_time/60:.1f} minutes")
    print(f"Expected speedup: {max_workers:.1f}x")
    
    # Confirm
    confirm = input("\nContinue with parallel comparison? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Parallel comparison cancelled.")
        return
    
    # Run parallel comparison
    results = runner.run_parallel_comparison(strategies=strategies_to_test, max_epochs=10)
    
    print(f"\n✅ Parallel comparison completed!")
    print(f"Check the results in: {runner.results_dir}/")

if __name__ == "__main__":
    main()

