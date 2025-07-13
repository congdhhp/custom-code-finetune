# Configuration Examples

This document provides various configuration examples for different use cases and hardware setups.

## Hardware-Specific Configurations

### RTX 3050 Ti (4GB VRAM) - Memory Optimized

**model_config.yaml**:
```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  trust_remote_code: true
  use_cache: false
  use_flash_attention_2: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 8
  alpha: 32
  dropout: 0.0
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"

memory:
  gradient_checkpointing: true
  device_map: "auto"
```

**training_config.yaml**:
```yaml
training:
  max_steps: 1500
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 3e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 75
  weight_decay: 0.01
  bf16: true
  fp16: false
  
  eval_steps: 300
  save_steps: 300
  save_total_limit: 3
  
  logging_steps: 25
  dataloader_num_workers: 2

data:
  seq_length: 1024  # Reduced for memory
  fim_rate: 0.3
```

### RTX 4060/4070 (8-12GB VRAM) - Balanced

**model_config.yaml**:
```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  trust_remote_code: true
  use_flash_attention_2: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: false

lora:
  r: 16
  alpha: 64
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**training_config.yaml**:
```yaml
training:
  max_steps: 2500
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-4
  
  eval_steps: 200
  save_steps: 200
  save_total_limit: 5

data:
  seq_length: 2048
  fim_rate: 0.5
```

### High-End GPU (16GB+ VRAM) - Performance

**model_config.yaml**:
```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  trust_remote_code: true
  use_flash_attention_2: true

quantization:
  load_in_4bit: false  # No quantization needed
  load_in_8bit: false

lora:
  r: 32
  alpha: 128
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**training_config.yaml**:
```yaml
training:
  max_steps: 3000
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-4
  
  eval_steps: 150
  save_steps: 150

data:
  seq_length: 4096
  fim_rate: 0.5
```

## Use Case Configurations

### Quick Experimentation

For rapid prototyping and testing:

```yaml
training:
  max_steps: 500
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 1e-3
  
  eval_steps: 100
  save_steps: 100
  
  early_stopping_patience: 2

data:
  seq_length: 512
  validation_split: 0.2
```

### Production Training

For final model training:

```yaml
training:
  max_steps: 5000
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 3e-4
  lr_scheduler_type: "cosine"
  warmup_steps: 250
  
  eval_steps: 250
  save_steps: 250
  save_total_limit: 10
  
  early_stopping_patience: 5
  early_stopping_threshold: 0.001

monitoring:
  report_to: ["tensorboard", "wandb"]
  wandb_project: "swtbot-production"
```

### Domain-Specific Training

For specific SWTBot use cases:

**data_config.yaml**:
```yaml
processing:
  swtbot_keywords:
    - "SWTBot"
    - "button"
    - "click"
    - "text"
    - "select"
    - "tree"
    - "table"
    - "menu"
    - "dialog"
    - "view"
    - "editor"
    - "shell"
  min_swtbot_keywords: 3
  min_lines_per_file: 15

dataset:
  augmentation:
    enabled: true
    methods:
      - "variable_renaming"
      - "comment_removal"
    augmentation_rate: 0.3
```

## Generation Configurations

### Conservative Generation

For reliable, safe code generation:

```python
generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.3,
    "top_k": 20,
    "top_p": 0.85,
    "repetition_penalty": 1.2,
    "do_sample": True
}
```

### Creative Generation

For diverse, creative outputs:

```python
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "do_sample": True
}
```

### Deterministic Generation

For consistent, reproducible outputs:

```python
generation_config = {
    "max_new_tokens": 384,
    "temperature": 0.1,
    "top_k": 10,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "do_sample": False  # Greedy decoding
}
```

## Environment-Specific Configurations

### Development Environment

```yaml
# Faster iteration, less resource usage
training:
  max_steps: 1000
  logging_steps: 10
  eval_steps: 100
  save_steps: 100

monitoring:
  log_level: "DEBUG"
  disable_tqdm: false
```

### CI/CD Environment

```yaml
# Automated testing and validation
training:
  max_steps: 200  # Quick validation
  per_device_train_batch_size: 1
  
validation:
  validation_examples: 100
  
monitoring:
  report_to: []  # No external logging
  log_level: "WARNING"
```

### Research Environment

```yaml
# Comprehensive logging and experimentation
training:
  max_steps: 10000
  
monitoring:
  report_to: ["tensorboard", "wandb"]
  include_tokens_per_second: true
  include_num_input_tokens_seen: true
  
checkpointing:
  save_on_each_node: true
  
evaluation:
  eval_accumulation_steps: 10
```

## Custom Data Configurations

### Large Dataset

```yaml
collection:
  clone_depth: null  # Full clone
  max_file_size_mb: 5
  
processing:
  min_lines_per_file: 20
  max_lines_per_file: 2000
  
dataset:
  train_split: 0.95
  validation_split: 0.05
  
cache:
  enabled: true
  cache_processed_files: true
```

### Small Dataset

```yaml
collection:
  clone_depth: 1
  max_file_size_mb: 1
  
processing:
  min_lines_per_file: 5
  min_swtbot_keywords: 1
  
dataset:
  train_split: 0.8
  validation_split: 0.2
  
  augmentation:
    enabled: true
    augmentation_rate: 0.5  # More augmentation for small datasets
```

## Multi-GPU Configurations

### Data Parallel Training

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  dataloader_num_workers: 8
  
memory:
  device_map: null  # Let PyTorch handle distribution
  
distributed:
  ddp_find_unused_parameters: false
  ddp_bucket_cap_mb: 25
```

### Model Parallel Training

```yaml
memory:
  device_map: "auto"  # Automatic model parallelism
  
model:
  low_cpu_mem_usage: true
  
training:
  remove_unused_columns: false
  dataloader_pin_memory: false
```

## Debugging Configurations

### Memory Debugging

```yaml
training:
  dataloader_num_workers: 0  # Single-threaded
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  
monitoring:
  log_level: "DEBUG"
  
memory:
  gradient_checkpointing: false  # Easier debugging
```

### Performance Profiling

```yaml
training:
  max_steps: 100  # Short run for profiling
  logging_steps: 1
  
monitoring:
  include_tokens_per_second: true
  profile: true
  
evaluation:
  evaluation_strategy: "no"  # Skip evaluation for profiling
```
