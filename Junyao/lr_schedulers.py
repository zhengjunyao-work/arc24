#!/usr/bin/env python3
"""
Comprehensive Learning Rate Scheduler Module
Includes warmup, multiple scheduler types, and monitoring
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

class WarmupScheduler:
    """Learning rate warmup scheduler"""
    
    def __init__(self, optimizer, warmup_epochs: int, method: str = 'linear', 
                 base_lr: float = None, target_lr: float = None):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            method: 'linear', 'exponential', 'cosine'
            base_lr: Base learning rate (if None, uses optimizer's lr)
            target_lr: Target learning rate (if None, uses optimizer's lr)
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.method = method
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.target_lr = target_lr or optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        
    def step(self, epoch: int = None):
        """Update learning rate for warmup"""
        if epoch is not None:
            self.current_epoch = epoch
            
        if self.current_epoch >= self.warmup_epochs:
            return
            
        # Calculate warmup factor
        if self.method == 'linear':
            factor = self.current_epoch / self.warmup_epochs
        elif self.method == 'exponential':
            factor = (np.exp(self.current_epoch / self.warmup_epochs) - 1) / (np.e - 1)
        elif self.method == 'cosine':
            factor = 0.5 * (1 + np.cos(np.pi * (1 - self.current_epoch / self.warmup_epochs)))
        else:
            raise ValueError(f"Unknown warmup method: {self.method}")
            
        # Set learning rate
        lr = self.base_lr + factor * (self.target_lr - self.base_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.current_epoch += 1

class AdaptiveLRScheduler:
    """Comprehensive adaptive learning rate scheduler with monitoring"""
    
    def __init__(self, optimizer, scheduler_type: str, scheduler_params: Dict,
                 use_warmup: bool = False, warmup_epochs: int = 0, 
                 warmup_method: str = 'linear', monitor_lr: bool = True):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler
            scheduler_params: Parameters for the scheduler
            use_warmup: Whether to use warmup
            warmup_epochs: Number of warmup epochs
            warmup_method: Warmup method
            monitor_lr: Whether to monitor learning rate
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.monitor_lr = monitor_lr
        self.lr_history = []
        self.epoch_history = []
        
        # Create warmup scheduler if needed
        if use_warmup and warmup_epochs > 0:
            self.warmup_scheduler = WarmupScheduler(
                optimizer, warmup_epochs, warmup_method
            )
        else:
            self.warmup_scheduler = None
            
        # Create main scheduler
        self.scheduler = self._create_scheduler(scheduler_type, scheduler_params)
        
    def _create_scheduler(self, scheduler_type: str, params: Dict):
        """Create the specified scheduler"""
        if scheduler_type == 'reduce_lr_on_plateau':
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-6),
                verbose=params.get('verbose', True),
                threshold=params.get('threshold', 1e-4),
                threshold_mode=params.get('threshold_mode', 'rel')
            )
            
        elif scheduler_type == 'cosine':
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get('T_max', 50),
                eta_min=params.get('eta_min', 1e-6),
                last_epoch=params.get('last_epoch', -1)
            )
            
        elif scheduler_type == 'step':
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get('step_size', 10),
                gamma=params.get('gamma', 0.5),
                last_epoch=params.get('last_epoch', -1)
            )
            
        elif scheduler_type == 'exponential':
            return lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=params.get('gamma', 0.95),
                last_epoch=params.get('last_epoch', -1)
            )
            
        elif scheduler_type == 'one_cycle':
            # OneCycleLR requires total_steps, which we'll set later
            return None
            
        elif scheduler_type == 'none':
            return None
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def setup_one_cycle(self, total_steps: int, epochs: int, steps_per_epoch: int):
        """Setup OneCycleLR scheduler (called after dataloader is created)"""
        if self.scheduler_type == 'one_cycle':
            params = self.scheduler_params
            self.scheduler = lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=params.get('max_lr', 1e-2),
                total_steps=total_steps,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=params.get('pct_start', 0.3),
                anneal_strategy=params.get('anneal_strategy', 'cos'),
                cycle_momentum=params.get('cycle_momentum', True),
                base_momentum=params.get('base_momentum', 0.85),
                max_momentum=params.get('max_momentum', 0.95),
                div_factor=params.get('div_factor', 25.0),
                final_div_factor=params.get('final_div_factor', 1e4)
            )
    
    def step(self, epoch: int = None, metrics: Optional[float] = None):
        """Step the scheduler"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Handle warmup
        if self.warmup_scheduler and epoch is not None and epoch < self.warmup_epochs:
            self.warmup_scheduler.step(epoch)
            if self.monitor_lr:
                self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                self.epoch_history.append(epoch)
            return
            
        # Step main scheduler
        if self.scheduler is not None:
            if self.scheduler_type == 'reduce_lr_on_plateau':
                if metrics is not None:
                    self.scheduler.step(metrics)
            else:
                self.scheduler.step()
                
        # Monitor learning rate
        if self.monitor_lr:
            self.lr_history.append(self.optimizer.param_groups[0]['lr'])
            if epoch is not None:
                self.epoch_history.append(epoch)
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_history(self) -> tuple:
        """Get learning rate history"""
        return self.epoch_history, self.lr_history
    
    def plot_lr_history(self, save_path: str = None):
        """Plot learning rate history"""
        if not self.lr_history:
            print("No learning rate history to plot")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Main plot
        plt.subplot(2, 2, 1)
        plt.plot(self.epoch_history, self.lr_history, 'b-', linewidth=2)
        plt.title('Learning Rate History', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Learning rate changes
        plt.subplot(2, 2, 2)
        lr_changes = np.diff(self.lr_history)
        plt.plot(self.epoch_history[1:], lr_changes, 'r-', linewidth=2)
        plt.title('Learning Rate Changes', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Δ Learning Rate')
        plt.grid(True, alpha=0.3)
        
        # Learning rate distribution
        plt.subplot(2, 2, 3)
        plt.hist(self.lr_history, bins=20, alpha=0.7, color='green')
        plt.title('Learning Rate Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Learning Rate')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""
        Learning Rate Statistics:
        
        Initial LR: {self.lr_history[0]:.2e}
        Final LR: {self.lr_history[-1]:.2e}
        Min LR: {min(self.lr_history):.2e}
        Max LR: {max(self.lr_history):.2e}
        Mean LR: {np.mean(self.lr_history):.2e}
        Std LR: {np.std(self.lr_history):.2e}
        
        Scheduler: {self.scheduler_type}
        Warmup: {'Yes' if self.use_warmup else 'No'}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning rate history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def create_adaptive_scheduler(optimizer, config: Dict) -> AdaptiveLRScheduler:
    """Factory function to create adaptive scheduler from config"""
    return AdaptiveLRScheduler(
        optimizer=optimizer,
        scheduler_type=config.get('LR_SCHEDULER_TYPE', 'reduce_lr_on_plateau'),
        scheduler_params=config.get('LR_SCHEDULER_PARAMS', {}),
        use_warmup=config.get('USE_WARMUP', False),
        warmup_epochs=config.get('WARMUP_EPOCHS', 0),
        warmup_method=config.get('WARMUP_METHOD', 'linear'),
        monitor_lr=config.get('MONITOR_LR', True)
    )
