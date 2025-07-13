"""
Custom training callbacks for SWTBot fine-tuning.
Includes checkpoint management, memory monitoring, and logging callbacks.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import wandb
import psutil

try:
    from ..model.quantization import log_memory_usage, optimize_memory_usage
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.quantization import log_memory_usage, optimize_memory_usage


class CheckpointCallback(TrainerCallback):
    """Custom callback for enhanced checkpoint management."""
    
    def __init__(self, save_steps: int, output_dir: str):
        """
        Initialize checkpoint callback.
        
        Args:
            save_steps: Steps between checkpoints
            output_dir: Output directory for checkpoints
        """
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        # Check if we should save a checkpoint
        if state.global_step % self.save_steps == 0:
            checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
            self.logger.info(f"Saving checkpoint at step {state.global_step}")
            
            # Save additional metadata
            self._save_checkpoint_metadata(checkpoint_dir, state)
    
    def _save_checkpoint_metadata(self, checkpoint_dir: Path, state: TrainerState):
        """Save additional metadata with checkpoint."""
        import json
        
        metadata = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "train_loss": state.log_history[-1].get("train_loss") if state.log_history else None,
            "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None,
            "timestamp": time.time(),
        }
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = checkpoint_dir / "training_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


class MemoryCallback(TrainerCallback):
    """Callback for monitoring and optimizing memory usage."""
    
    def __init__(self, log_frequency: int = 100):
        """
        Initialize memory callback.
        
        Args:
            log_frequency: Steps between memory logging
        """
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(__name__)
        self.step_count = 0
        
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, **kwargs):
        """Called at the beginning of each training step."""
        self.step_count += 1
        
        # Log memory usage periodically
        if self.step_count % self.log_frequency == 0:
            log_memory_usage()
            
        # Optimize memory usage periodically
        if self.step_count % (self.log_frequency * 2) == 0:
            optimize_memory_usage()
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called at the end of each epoch."""
        self.logger.info(f"Epoch {state.epoch} completed")
        log_memory_usage()
        optimize_memory_usage()


class WandbCallback(TrainerCallback):
    """Enhanced Weights & Biases logging callback."""
    
    def __init__(self):
        """Initialize wandb callback."""
        self.logger = logging.getLogger(__name__)
        
    def on_log(self, args: TrainingArguments, state: TrainerState, 
              control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called when logging metrics."""
        if logs is None:
            return
            
        # Add custom metrics
        custom_logs = logs.copy()
        
        # Add memory metrics if available
        if torch.cuda.is_available():
            custom_logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            custom_logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        # Add system memory
        memory_info = psutil.virtual_memory()
        custom_logs["system_memory_percent"] = memory_info.percent
        
        # Log to wandb
        try:
            wandb.log(custom_logs, step=state.global_step)
        except Exception as e:
            self.logger.warning(f"Failed to log to wandb: {e}")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        # Log training configuration
        try:
            wandb.config.update({
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "save_steps": args.save_steps,
                "eval_steps": args.eval_steps,
            })
        except Exception as e:
            self.logger.warning(f"Failed to update wandb config: {e}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called at the end of training."""
        self.logger.info("Training completed, finalizing wandb logging")


class EvaluationCallback(TrainerCallback):
    """Custom callback for enhanced evaluation logging."""
    
    def __init__(self, eval_frequency: int = None):
        """
        Initialize evaluation callback.
        
        Args:
            eval_frequency: Steps between evaluations (if different from args.eval_steps)
        """
        self.eval_frequency = eval_frequency
        self.logger = logging.getLogger(__name__)
        self.best_eval_loss = float('inf')
        self.best_step = 0
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
                   control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called after evaluation."""
        if logs is None:
            return
            
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_step = state.global_step
                self.logger.info(f"New best evaluation loss: {eval_loss:.4f} at step {state.global_step}")
            
            # Log evaluation metrics
            self.logger.info(f"Evaluation at step {state.global_step}:")
            for key, value in logs.items():
                if key.startswith("eval_"):
                    self.logger.info(f"  {key}: {value:.4f}")


class ProgressCallback(TrainerCallback):
    """Callback for enhanced progress reporting."""
    
    def __init__(self, log_frequency: int = 50):
        """
        Initialize progress callback.
        
        Args:
            log_frequency: Steps between progress logs
        """
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.logger.info(f"Training started - Target steps: {args.max_steps}")
        
    def on_log(self, args: TrainingArguments, state: TrainerState, 
              control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called when logging metrics."""
        if logs is None or state.global_step % self.log_frequency != 0:
            return
            
        # Calculate progress
        progress = state.global_step / args.max_steps * 100
        elapsed_time = time.time() - self.start_time
        
        # Estimate remaining time
        if state.global_step > 0:
            time_per_step = elapsed_time / state.global_step
            remaining_steps = args.max_steps - state.global_step
            estimated_remaining = time_per_step * remaining_steps
            
            self.logger.info(
                f"Progress: {progress:.1f}% ({state.global_step}/{args.max_steps}) - "
                f"Elapsed: {elapsed_time/3600:.1f}h - "
                f"Remaining: {estimated_remaining/3600:.1f}h"
            )
            
            # Log current metrics
            if "train_loss" in logs:
                self.logger.info(f"Current train loss: {logs['train_loss']:.4f}")
            if "learning_rate" in logs:
                self.logger.info(f"Current learning rate: {logs['learning_rate']:.2e}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
