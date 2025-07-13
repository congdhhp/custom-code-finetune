#!/usr/bin/env python3
"""
Main training script for SWTBot Fine-tuning Pipeline.
Orchestrates the complete training process.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, merge_configs, validate_config
from utils.logging import setup_logging
from data.dataset import ConstantLengthDataset
from training.trainer import SWTBotTrainer
from training.utils import validate_training_config, create_run_name
from model.quantization import check_memory_requirements


def setup_datasets(config: dict, tokenizer) -> tuple:
    """
    Setup training and evaluation datasets.
    
    Args:
        config: Complete configuration dictionary
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger = logging.getLogger(__name__)
    
    # Load processed data
    data_config = config.get("data", {})
    processed_data_file = Path(data_config.get("output", {}).get("processed_data_dir", "data/processed")) / "processed_java_files.jsonl"
    
    if not processed_data_file.exists():
        raise FileNotFoundError(f"Processed data file not found: {processed_data_file}")
    
    logger.info(f"Loading data from {processed_data_file}")
    
    # Load data
    data_examples = []
    with open(processed_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            data_examples.append(example)
    
    logger.info(f"Loaded {len(data_examples)} examples")
    
    # Split data
    validation_split = config.get("dataset", {}).get("validation_split", 0.1)
    split_idx = int(len(data_examples) * (1 - validation_split))
    
    train_examples = data_examples[:split_idx]
    eval_examples = data_examples[split_idx:]
    
    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Eval examples: {len(eval_examples)}")
    
    # Create datasets
    dataset_config = config.get("dataset", {})
    
    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=train_examples,
        infinite=True,
        seq_length=dataset_config.get("seq_length", 2048),
        chars_per_token=dataset_config.get("chars_per_token", 2.5),
        content_field="content",
        fim_rate=dataset_config.get("fim_rate", 0.5),
        fim_spm_rate=dataset_config.get("fim_spm_rate", 0.5),
        seed=config.get("training", {}).get("seed", 42),
    )
    
    eval_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=eval_examples,
        infinite=False,
        seq_length=dataset_config.get("seq_length", 2048),
        chars_per_token=dataset_config.get("chars_per_token", 2.5),
        content_field="content",
        fim_rate=0.0,  # No FIM for evaluation
        fim_spm_rate=0.0,
        seed=config.get("training", {}).get("seed", 42),
    )
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SWTBot fine-tuned model")
    parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--training-config",
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override per-device batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--wandb-project",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.log_level, "logs/training.log")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SWTBot fine-tuning training")
    logger.info("=" * 60)
    
    try:
        # Load configurations
        logger.info("Loading configurations")
        model_config = load_config(args.model_config)
        training_config = load_config(args.training_config)
        data_config = load_config(args.data_config)

        # Merge configurations properly
        config = {
            "model": model_config.get("model", {}),
            "quantization": model_config.get("quantization", {}),
            "lora": model_config.get("lora", {}),
            "tokenizer": model_config.get("tokenizer", {}),
            "memory": model_config.get("memory", {}),
            "training": training_config.get("training", {}),
            "data": data_config.get("data", {}),
            "dataset": data_config.get("dataset", {}),
        }

        # Debug: Print loaded configuration
        logger.info(f"Model name: {config['model'].get('name', 'NOT FOUND')}")
        logger.info(f"Training max_steps: {config['training'].get('max_steps', 'NOT FOUND')}")
        
        # Apply command line overrides
        if args.output_dir:
            config["training"]["output_dir"] = args.output_dir
        if args.max_steps:
            config["training"]["max_steps"] = args.max_steps
        if args.batch_size:
            config["training"]["per_device_train_batch_size"] = args.batch_size
        if args.learning_rate:
            config["training"]["learning_rate"] = args.learning_rate
        if args.wandb_project:
            config["training"]["wandb_project"] = args.wandb_project
            if "wandb" not in config["training"].get("report_to", []):
                config["training"]["report_to"] = config["training"].get("report_to", []) + ["wandb"]
        
        # Validate configurations
        logger.info("Validating configurations")
        validate_training_config(config["training"])
        
        # Check memory requirements
        model_name = config["model"].get("name", "meta-llama/Llama-3.2-1B-Instruct")
        seq_length = config.get("dataset", {}).get("seq_length", 2048)
        batch_size = config["training"].get("per_device_train_batch_size", 1)
        use_4bit = config.get("quantization", {}).get("load_in_4bit", True)
        
        if not check_memory_requirements(model_name, seq_length, batch_size, use_4bit):
            logger.error("Memory requirements check failed")
            if not args.dry_run:
                sys.exit(1)
        
        # Create run name
        run_name = create_run_name(config)
        logger.info(f"Run name: {run_name}")
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return
        
        # Initialize trainer
        logger.info("Initializing trainer")
        trainer = SWTBotTrainer(config)
        trainer.initialize()
        
        # Setup datasets
        logger.info("Setting up datasets")
        train_dataset, eval_dataset = setup_datasets(config, trainer.tokenizer)
        trainer.prepare_datasets(train_dataset, eval_dataset)
        
        # Setup trainer
        trainer.setup_trainer()
        
        # Save configuration
        output_dir = Path(config["training"]["output_dir"])
        config_file = output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config_file}")
        
        # Start training
        logger.info("Starting training process")
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Log training results
        logger.info("Training completed successfully!")
        logger.info("Training statistics:")
        for key, value in train_result.metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Run final evaluation
        logger.info("Running final evaluation")
        eval_result = trainer.evaluate()
        if eval_result:
            logger.info("Final evaluation results:")
            for key, value in eval_result.items():
                logger.info(f"  {key}: {value}")
        
        # Save final statistics
        stats_file = output_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "train_result": train_result.metrics,
                "eval_result": eval_result,
                "config": config
            }, f, indent=2, default=str)
        
        logger.info(f"Training statistics saved to {stats_file}")
        logger.info("Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
