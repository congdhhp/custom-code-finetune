#!/usr/bin/env python3
"""
Minimal training script for SWTBot Fine-tuning.
Compatible with older transformers versions.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
        set_seed
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    logger.info("✅ All packages imported successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.error("Please install: pip install torch transformers peft bitsandbytes")
    sys.exit(1)


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for training."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_training_data():
    """Load training data from processed files."""
    data_file = Path("data/processed/processed_java_files.jsonl")
    
    if not data_file.exists():
        logger.error(f"Training data not found: {data_file}")
        logger.error("Run: python create_sample_data.py")
        sys.exit(1)
    
    # Load data
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item["content"])
    
    logger.info(f"Loaded {len(data)} training examples")
    return data


def tokenize_data(data, tokenizer, max_length=512):
    """Tokenize the training data."""
    tokenized_data = []
    
    logger.info("Tokenizing data...")
    for content in data:
        # Tokenize
        encoding = tokenizer(
            content,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        tokenized_data.append({
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        })
    
    return tokenized_data


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Minimal SWTBot training")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=500, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--output-dir", default="./minimal_model", help="Output directory")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    
    args = parser.parse_args()
    
    logger.info("Starting minimal SWTBot training")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        device_map = "auto"
        use_quantization = True
    else:
        logger.warning("⚠️  No GPU detected, using CPU")
        device_map = None
        use_quantization = False
    
    # Set seed
    set_seed(42)
    
    # Model configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    logger.info(f"Loading model: {model_name}")
    
    # Setup quantization
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit quantization enabled")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_quantization else torch.float32,
    )
    
    # Prepare for training
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare data
    logger.info("Loading training data...")
    raw_data = load_training_data()
    
    # Tokenize data
    tokenized_data = tokenize_data(raw_data, tokenizer, args.max_length)
    
    # Split data
    split_idx = int(len(tokenized_data) * 0.9)
    train_data = tokenized_data[:split_idx]
    eval_data = tokenized_data[split_idx:]
    
    # Create datasets
    train_dataset = SimpleDataset(train_data)
    eval_dataset = SimpleDataset(eval_data)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training arguments (minimal, compatible version)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=25,
        save_steps=100,
        eval_steps=100,
        load_best_model_at_end=False,  # Disable to avoid strategy mismatch
        bf16=use_quantization,
        fp16=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=42,
        data_seed=42,
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("✅ Training completed successfully!")
        
        # Save final model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Print results
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")
        
        # Save training info
        info = {
            "model_name": model_name,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "final_loss": train_result.training_loss,
            "training_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
        }
        
        with open(Path(args.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
