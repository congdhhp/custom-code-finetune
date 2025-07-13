"""
Main training module for SWTBot fine-tuning.
Implements the training pipeline with checkpoint saving and monitoring.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

try:
    from ..model.config import ModelConfig
    from ..model.quantization import optimize_memory_usage, log_memory_usage
    from ..data.dataset import ConstantLengthDataset
    from .callbacks import CheckpointCallback, MemoryCallback, WandbCallback
    from .utils import setup_training_args, calculate_steps
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.config import ModelConfig
    from model.quantization import optimize_memory_usage, log_memory_usage
    from data.dataset import ConstantLengthDataset
    from training.callbacks import CheckpointCallback, MemoryCallback, WandbCallback
    from training.utils import setup_training_args, calculate_steps


class SWTBotTrainer:
    """Main trainer class for SWTBot fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWTBot trainer.
        
        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})
        self.data_config = config.get("data", {})
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_manager = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Training state
        self.is_initialized = False
        self.training_stats = {}
        
        # Set random seed for reproducibility
        seed = self.training_config.get("seed", 42)
        set_seed(seed)
        self.logger.info(f"Random seed set to {seed}")
    
    def initialize(self):
        """Initialize the trainer with model and data."""
        if self.is_initialized:
            self.logger.warning("Trainer already initialized")
            return
        
        self.logger.info("Initializing SWTBot trainer")
        
        # Setup model
        self._setup_model()
        
        # Setup training arguments
        self._setup_training_args()
        
        # Initialize wandb if configured
        self._setup_wandb()
        
        self.is_initialized = True
        self.logger.info("Trainer initialization complete")
    
    def _setup_model(self):
        """Setup model and tokenizer."""
        self.logger.info("Setting up model and tokenizer")
        
        # Create model configuration manager
        self.model_manager = ModelConfig(self.config)
        
        # Setup complete model pipeline
        self.model, self.tokenizer = self.model_manager.setup_complete_model()
        
        # Log model information
        model_info = self.model_manager.get_model_info()
        self.logger.info(f"Model loaded: {model_info['model_name']}")
        self.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,} "
                        f"({model_info['trainable_percentage']:.2f}%)")
        
        # Log memory usage
        log_memory_usage()
    
    def _setup_training_args(self):
        """Setup training arguments."""
        self.logger.info("Setting up training arguments")
        
        # Create training arguments
        self.training_args = setup_training_args(self.training_config)
        
        self.logger.info(f"Training arguments configured:")
        self.logger.info(f"  Output directory: {self.training_args.output_dir}")
        self.logger.info(f"  Max steps: {self.training_args.max_steps}")
        self.logger.info(f"  Batch size: {self.training_args.per_device_train_batch_size}")
        self.logger.info(f"  Learning rate: {self.training_args.learning_rate}")
        self.logger.info(f"  Save steps: {self.training_args.save_steps}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging if configured."""
        if "wandb" in self.training_config.get("report_to", []):
            try:
                # Initialize wandb
                wandb.init(
                    project=self.training_config.get("wandb_project", "swtbot-finetune"),
                    name=self.training_config.get("wandb_run_name", "swtbot-training"),
                    config=self.config,
                    tags=["swtbot", "llama", "fine-tuning"]
                )
                self.logger.info("Wandb initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
    
    def prepare_datasets(self, train_dataset, eval_dataset=None):
        """
        Prepare training and evaluation datasets.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        self.logger.info("Preparing datasets")
        
        # Store datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Log dataset information
        if hasattr(train_dataset, '__len__'):
            self.logger.info(f"Training dataset size: {len(train_dataset)}")
        else:
            self.logger.info("Training dataset: streaming/iterable")
        
        if eval_dataset:
            if hasattr(eval_dataset, '__len__'):
                self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            else:
                self.logger.info("Evaluation dataset: streaming/iterable")
    
    def setup_trainer(self):
        """Setup the Hugging Face Trainer."""
        if not self.is_initialized:
            raise RuntimeError("Trainer not initialized. Call initialize() first.")
        
        if not hasattr(self, 'train_dataset'):
            raise RuntimeError("Datasets not prepared. Call prepare_datasets() first.")
        
        self.logger.info("Setting up Hugging Face Trainer")
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        
        self.logger.info("Trainer setup complete")
    
    def _setup_callbacks(self) -> List:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping callback (only if evaluation is enabled)
        if (self.training_config.get("early_stopping_patience") and
            self.training_args.evaluation_strategy != "no"):
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.training_config["early_stopping_patience"],
                early_stopping_threshold=self.training_config.get("early_stopping_threshold", 0.001)
            )
            callbacks.append(early_stopping)
            self.logger.info("Early stopping callback added")
        elif self.training_config.get("early_stopping_patience"):
            self.logger.warning("Early stopping disabled: evaluation strategy is 'no'")
        
        # Custom checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_steps=self.training_args.save_steps,
            output_dir=self.training_args.output_dir
        )
        callbacks.append(checkpoint_callback)
        
        # Memory monitoring callback
        memory_callback = MemoryCallback()
        callbacks.append(memory_callback)
        
        # Wandb callback if configured
        if "wandb" in self.training_config.get("report_to", []):
            wandb_callback = WandbCallback()
            callbacks.append(wandb_callback)
        
        return callbacks
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Start the training process.
        
        Args:
            resume_from_checkpoint: Optional checkpoint path to resume from
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")
        
        self.logger.info("Starting training")
        
        # Check for existing checkpoints if resume path not specified
        if resume_from_checkpoint is None:
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint:
                self.logger.info(f"Found existing checkpoint: {last_checkpoint}")
                resume_from_checkpoint = last_checkpoint
        
        # Optimize memory before training
        optimize_memory_usage()
        
        try:
            # Start training
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save training statistics
            self.training_stats = train_result.metrics
            self.logger.info("Training completed successfully")
            
            # Save final model
            self._save_final_model()
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            if "wandb" in self.training_config.get("report_to", []):
                wandb.finish()
    
    def _save_final_model(self):
        """Save the final trained model."""
        final_model_dir = Path(self.training_args.output_dir) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using the model manager
        self.model_manager.save_model(str(final_model_dir))
        
        # Save training configuration
        config_file = final_model_dir / "training_config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        self.logger.info(f"Final model saved to {final_model_dir}")
    
    def evaluate(self):
        """Run evaluation on the evaluation dataset."""
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first.")
        
        if self.eval_dataset is None:
            self.logger.warning("No evaluation dataset provided")
            return None
        
        self.logger.info("Running evaluation")
        eval_result = self.trainer.evaluate()
        
        self.logger.info(f"Evaluation results: {eval_result}")
        return eval_result
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats
    
    def save_checkpoint(self, checkpoint_dir: str):
        """
        Save a training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not setup")
        
        self.trainer.save_model(checkpoint_dir)
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
