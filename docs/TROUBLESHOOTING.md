# Troubleshooting Guide

This guide helps resolve common issues when using the SWTBot fine-tuning pipeline.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Memory Issues](#memory-issues)
3. [Training Problems](#training-problems)
4. [Data Collection Issues](#data-collection-issues)
5. [Generation Issues](#generation-issues)
6. [Performance Issues](#performance-issues)
7. [Configuration Errors](#configuration-errors)

## Installation Issues

### Python Version Compatibility

**Problem**: `Error: Python 3.8 or higher is required`

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.8+ if needed
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev

# On Windows: Download from python.org
# On macOS: Use Homebrew
brew install python@3.8
```

### CUDA/GPU Issues

**Problem**: `CUDA out of memory` or `No GPU detected`

**Solutions**:

1. **Check GPU availability**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

2. **Install correct PyTorch version**:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Dependency Conflicts

**Problem**: Package version conflicts

**Solution**:
```bash
# Create fresh environment
python -m venv fresh_venv
source fresh_venv/bin/activate  # Linux/Mac
# or
fresh_venv\Scripts\activate     # Windows

# Install dependencies one by one
pip install torch transformers datasets
pip install peft bitsandbytes
pip install -r requirements.txt
```

## Memory Issues

### GPU Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
```bash
python scripts/train.py --batch-size 1
```

2. **Increase gradient accumulation**:
```yaml
# In configs/training_config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

3. **Use more aggressive quantization**:
```yaml
# In configs/model_config.yaml
quantization:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
```

4. **Reduce sequence length**:
```yaml
# In configs/data_config.yaml
dataset:
  seq_length: 1024  # Instead of 2048
```

### System Memory Issues

**Problem**: System runs out of RAM

**Solutions**:

1. **Reduce data workers**:
```yaml
# In configs/training_config.yaml
training:
  dataloader_num_workers: 2  # Instead of 4
```

2. **Use streaming datasets**:
```python
# The pipeline already uses streaming by default
# Ensure infinite=True for training dataset
```

3. **Clear cache regularly**:
```python
import torch
import gc

# Add to training loop
if step % 100 == 0:
    torch.cuda.empty_cache()
    gc.collect()
```

## Training Problems

### Training Stops/Crashes

**Problem**: Training stops unexpectedly

**Solutions**:

1. **Check logs**:
```bash
tail -f logs/training.log
```

2. **Resume from checkpoint**:
```bash
python scripts/train.py --resume-from-checkpoint models/checkpoints/checkpoint-1000
```

3. **Reduce complexity**:
```bash
python scripts/train.py \
    --max-steps 1000 \
    --batch-size 1 \
    --learning-rate 3e-4
```

### Loss Not Decreasing

**Problem**: Training loss remains high or increases

**Solutions**:

1. **Check learning rate**:
```yaml
# Try different learning rates
training:
  learning_rate: 3e-4  # Instead of 5e-4
```

2. **Verify data quality**:
```bash
# Check data processing stats
cat data/processed/processing_stats.json
```

3. **Adjust LoRA parameters**:
```yaml
# In configs/model_config.yaml
lora:
  r: 16      # Increase rank
  alpha: 64  # Increase alpha
```

### Evaluation Issues

**Problem**: Evaluation fails or gives poor results

**Solutions**:

1. **Check evaluation data**:
```bash
# Ensure evaluation dataset exists
ls data/processed/
```

2. **Adjust evaluation frequency**:
```yaml
training:
  eval_steps: 500  # Less frequent evaluation
```

## Data Collection Issues

### Repository Access

**Problem**: `Failed to clone repository`

**Solutions**:

1. **Check internet connection**
2. **Use different repositories**:
```yaml
# In configs/data_config.yaml
data_sources:
  repositories:
    - url: "https://github.com/your-repo/swtbot-code"
      name: "custom-swtbot"
```

3. **Manual data addition**:
```bash
# Place files in data/raw/manual/
mkdir -p data/raw/manual/files
# Copy your .java files here
python scripts/collect_data.py --skip-collection
```

### Processing Failures

**Problem**: `No files processed` or processing errors

**Solutions**:

1. **Check file filters**:
```yaml
# In configs/data_config.yaml
processing:
  min_swtbot_keywords: 1  # Reduce requirement
  min_lines_per_file: 5   # Reduce requirement
```

2. **Verify file content**:
```bash
# Check collected files
find data/raw -name "*.java" | head -5 | xargs head -20
```

## Generation Issues

### Poor Code Quality

**Problem**: Generated code is incomplete or incorrect

**Solutions**:

1. **Adjust generation parameters**:
```bash
python scripts/generate.py \
    --temperature 0.5 \
    --top-k 40 \
    --repetition-penalty 1.2
```

2. **Use better prompts**:
```bash
# Instead of: "test button"
# Use: "Create a SWTBot test that clicks a button with text 'Submit' and verifies the result"
```

3. **Train longer**:
```bash
python scripts/train.py --max-steps 3000
```

### Model Loading Errors

**Problem**: `Model not found` or loading failures

**Solutions**:

1. **Check model path**:
```bash
ls models/checkpoints/final_model/
# Should contain: config.json, pytorch_model.bin, tokenizer files
```

2. **Use checkpoint instead**:
```bash
python scripts/generate.py \
    --model-path models/checkpoints/checkpoint-2000
```

## Performance Issues

### Slow Training

**Problem**: Training is very slow

**Solutions**:

1. **Enable optimizations**:
```yaml
# In configs/model_config.yaml
model:
  use_flash_attention_2: true

memory:
  gradient_checkpointing: true
```

2. **Optimize data loading**:
```yaml
training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
```

3. **Use mixed precision**:
```yaml
training:
  bf16: true
  fp16: false
```

### Slow Generation

**Problem**: Code generation is slow

**Solutions**:

1. **Reduce max tokens**:
```bash
python scripts/generate.py \
    --max-new-tokens 256
```

2. **Use quantization**:
```python
# In generation config
generation_config = {
    "use_quantization": True
}
```

## Configuration Errors

### YAML Syntax Errors

**Problem**: `YAML parsing error`

**Solutions**:

1. **Check indentation**:
```yaml
# Correct:
training:
  max_steps: 2000
  batch_size: 2

# Incorrect:
training:
max_steps: 2000
  batch_size: 2
```

2. **Validate YAML**:
```python
import yaml
with open('configs/training_config.yaml') as f:
    config = yaml.safe_load(f)
    print("YAML is valid")
```

### Missing Configuration

**Problem**: `Configuration key missing`

**Solutions**:

1. **Use default configs**:
```bash
# Copy default configurations
cp configs/model_config.yaml configs/my_model_config.yaml
```

2. **Check required keys**:
```python
# The pipeline validates required keys automatically
# Check error messages for missing keys
```

## Getting Help

### Enable Debug Logging

```bash
python scripts/train.py --log-level DEBUG
python scripts/generate.py --log-level DEBUG
```

### Check System Resources

```bash
# GPU memory
nvidia-smi

# System memory
free -h

# Disk space
df -h
```

### Collect Information

When reporting issues, include:

1. **System information**:
```bash
python --version
pip list | grep torch
nvidia-smi
```

2. **Error logs**:
```bash
tail -50 logs/training.log
```

3. **Configuration files**:
```bash
cat configs/model_config.yaml
cat configs/training_config.yaml
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU memory insufficient | Reduce batch size, use quantization |
| `No module named 'transformers'` | Missing dependency | `pip install transformers` |
| `Model not found` | Incorrect model path | Check path, use absolute path |
| `YAML parsing error` | Invalid YAML syntax | Check indentation, quotes |
| `Permission denied` | File permissions | `chmod +x scripts/*.py` |

### Performance Benchmarks

**RTX 3050 Ti (4GB VRAM)**:
- Batch size: 1-2
- Training speed: ~0.5 steps/second
- Memory usage: ~3.5GB
- Recommended max_steps: 1500-2000

**RTX 4060 (8GB VRAM)**:
- Batch size: 2-4
- Training speed: ~1.0 steps/second
- Memory usage: ~6GB
- Recommended max_steps: 2000-3000

If your performance is significantly lower, check for:
- Background processes using GPU
- Incorrect CUDA installation
- Thermal throttling
- Power limit restrictions
