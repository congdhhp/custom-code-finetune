"""
Training utilities for SWTBot fine-tuning.
Helper functions for training setup and configuration.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from transformers import TrainingArguments


def setup_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """
    Setup training arguments from configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        TrainingArguments object
    """
    logger = logging.getLogger(__name__)
    
    # Extract configuration values with defaults and type conversion
    output_dir = config.get("output_dir", "models/checkpoints")
    max_steps = int(config.get("max_steps", 2000))
    num_train_epochs = int(config.get("num_train_epochs", 3))

    # Batch size and accumulation
    per_device_train_batch_size = int(config.get("per_device_train_batch_size", 2))
    per_device_eval_batch_size = int(config.get("per_device_eval_batch_size", 2))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 8))

    # Learning rate and scheduling
    learning_rate = float(config.get("learning_rate", 5e-4))
    lr_scheduler_type = config.get("lr_scheduler_type", "cosine")
    warmup_steps = int(config.get("warmup_steps", 100))
    weight_decay = float(config.get("weight_decay", 0.01))
    
    # Mixed precision
    fp16 = config.get("fp16", False)
    bf16 = config.get("bf16", True)
    
    # Evaluation and saving
    evaluation_strategy = config.get("evaluation_strategy", "steps")
    eval_steps = int(config.get("eval_steps", 200))
    save_strategy = config.get("save_strategy", "steps")
    save_steps = int(config.get("save_steps", 200))
    save_total_limit = int(config.get("save_total_limit", 5))

    # Logging
    logging_strategy = config.get("logging_strategy", "steps")
    logging_steps = int(config.get("logging_steps", 50))
    report_to = config.get("report_to", ["tensorboard"])

    # Data handling
    dataloader_drop_last = bool(config.get("dataloader_drop_last", True))
    dataloader_num_workers = int(config.get("dataloader_num_workers", 4))
    
    # Optimization
    optim = config.get("optim", "adamw_torch")
    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    adam_epsilon = float(config.get("adam_epsilon", 1e-8))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    # Reproducibility
    seed = int(config.get("seed", 42))
    data_seed = int(config.get("data_seed", 42))
    
    # Model selection
    load_best_model_at_end = config.get("load_best_model_at_end", True)
    metric_for_best_model = config.get("metric_for_best_model", "eval_loss")
    greater_is_better = config.get("greater_is_better", False)
    
    # Other settings
    overwrite_output_dir = config.get("overwrite_output_dir", True)
    include_tokens_per_second = config.get("include_tokens_per_second", True)
    include_num_input_tokens_seen = config.get("include_num_input_tokens_seen", True)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create TrainingArguments with compatibility handling
    training_args_dict = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "fp16": fp16,
        "bf16": bf16,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "logging_steps": logging_steps,
        "dataloader_drop_last": dataloader_drop_last,
        "dataloader_num_workers": dataloader_num_workers,
        "optim": optim,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "adam_epsilon": adam_epsilon,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "data_seed": data_seed,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "overwrite_output_dir": overwrite_output_dir,
        "gradient_checkpointing": True,  # Always enable for memory efficiency
    }

    # Add optional parameters that might not be supported in all versions
    optional_params = {
        "evaluation_strategy": evaluation_strategy,
        "save_strategy": save_strategy,
        "logging_strategy": logging_strategy,
        "report_to": report_to,
        "include_tokens_per_second": include_tokens_per_second,
        "include_num_input_tokens_seen": include_num_input_tokens_seen,
    }

    # Try to add optional parameters one by one
    for param_name, param_value in optional_params.items():
        try:
            # Test if parameter is supported by creating a dummy TrainingArguments
            test_args = {"output_dir": "test", param_name: param_value}
            TrainingArguments(**test_args)
            training_args_dict[param_name] = param_value
        except (TypeError, ValueError):
            logger.warning(f"Parameter '{param_name}' not supported in this transformers version, skipping")

    # Fix evaluation strategy compatibility
    if "evaluation_strategy" not in training_args_dict:
        # If evaluation_strategy is not supported, disable load_best_model_at_end
        if training_args_dict.get("load_best_model_at_end", False):
            logger.warning("Disabling load_best_model_at_end due to evaluation_strategy compatibility")
            training_args_dict["load_best_model_at_end"] = False
    else:
        # Ensure evaluation and save strategies match
        eval_strategy = training_args_dict.get("evaluation_strategy", "no")
        save_strategy = training_args_dict.get("save_strategy", "steps")

        if eval_strategy == "no" and training_args_dict.get("load_best_model_at_end", False):
            logger.warning("Setting evaluation_strategy to match save_strategy for load_best_model_at_end")
            training_args_dict["evaluation_strategy"] = save_strategy

    # Create final TrainingArguments
    training_args = TrainingArguments(**training_args_dict)
    
    logger.info("Training arguments configured successfully")
    return training_args


def calculate_steps(dataset_size: int, batch_size: int, gradient_accumulation_steps: int,
                   num_epochs: int) -> Tuple[int, int]:
    """
    Calculate training steps and steps per epoch.
    
    Args:
        dataset_size: Number of examples in dataset
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (total_steps, steps_per_epoch)
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    return total_steps, steps_per_epoch


def estimate_training_time(total_steps: int, steps_per_second: float = None) -> Dict[str, float]:
    """
    Estimate training time based on steps and performance.
    
    Args:
        total_steps: Total training steps
        steps_per_second: Estimated steps per second (if known)
        
    Returns:
        Dictionary with time estimates
    """
    if steps_per_second is None:
        # Conservative estimate for RTX 3050 Ti with quantization
        steps_per_second = 0.5  # Adjust based on actual performance
    
    total_seconds = total_steps / steps_per_second
    
    return {
        "total_steps": total_steps,
        "steps_per_second": steps_per_second,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "total_days": total_seconds / (3600 * 24),
    }


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration for common issues.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    logger = logging.getLogger(__name__)
    issues = []
    
    # Check batch size
    batch_size = config.get("per_device_train_batch_size", 2)
    try:
        batch_size = int(batch_size)
        if batch_size < 1:
            issues.append("Batch size must be at least 1")
    except (ValueError, TypeError):
        issues.append(f"Batch size {batch_size} is not a valid integer")
    
    # Check learning rate
    learning_rate = config.get("learning_rate", 5e-4)
    try:
        learning_rate = float(learning_rate)
        if learning_rate <= 0 or learning_rate > 1:
            issues.append(f"Learning rate {learning_rate} seems unreasonable")
    except (ValueError, TypeError):
        issues.append(f"Learning rate {learning_rate} is not a valid number")
    
    # Check save/eval steps
    try:
        save_steps = int(config.get("save_steps", 200))
        eval_steps = int(config.get("eval_steps", 200))
        max_steps = int(config.get("max_steps", 2000))

        if save_steps > max_steps:
            issues.append("Save steps greater than max steps")
        if eval_steps > max_steps:
            issues.append("Eval steps greater than max steps")
    except (ValueError, TypeError):
        issues.append("Invalid step values in configuration")
    
    # Check mixed precision settings
    fp16 = config.get("fp16", False)
    bf16 = config.get("bf16", True)
    if fp16 and bf16:
        issues.append("Both fp16 and bf16 are enabled - choose one")
    
    # Check gradient accumulation
    try:
        grad_acc_steps = int(config.get("gradient_accumulation_steps", 8))
        if grad_acc_steps < 1:
            issues.append("Gradient accumulation steps must be at least 1")
    except (ValueError, TypeError):
        issues.append("Invalid gradient accumulation steps")
    
    # Log issues
    if issues:
        logger.warning("Training configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("Training configuration validation passed")
    return True


def get_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get optimizer configuration from training config.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Optimizer configuration dictionary
    """
    return {
        "optim": config.get("optim", "adamw_torch"),
        "learning_rate": config.get("learning_rate", 5e-4),
        "weight_decay": config.get("weight_decay", 0.01),
        "adam_beta1": config.get("adam_beta1", 0.9),
        "adam_beta2": config.get("adam_beta2", 0.999),
        "adam_epsilon": config.get("adam_epsilon", 1e-8),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
    }


def get_scheduler_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get learning rate scheduler configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Scheduler configuration dictionary
    """
    return {
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "warmup_steps": config.get("warmup_steps", 100),
        "warmup_ratio": config.get("warmup_ratio", 0.0),
    }


def create_run_name(config: Dict[str, Any]) -> str:
    """
    Create a descriptive run name for logging.
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        Run name string
    """
    model_name = config.get("model", {}).get("name", "llama").split("/")[-1]
    batch_size = config.get("training", {}).get("per_device_train_batch_size", 2)
    learning_rate = config.get("training", {}).get("learning_rate", 5e-4)
    lora_r = config.get("lora", {}).get("r", 8)
    
    run_name = f"swtbot-{model_name}-bs{batch_size}-lr{learning_rate:.0e}-r{lora_r}"
    
    return run_name
