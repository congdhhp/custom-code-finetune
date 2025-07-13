# SWTBot Fine-tuning Pipeline - Usage Guide

This guide provides detailed instructions for using the SWTBot fine-tuning pipeline to train and use models for generating SWTBot test code.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Data Collection](#data-collection)
4. [Training](#training)
5. [Code Generation](#code-generation)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd custom-code-finetune

# Setup environment and install dependencies
python scripts/setup_env.py

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2. Data Collection

```bash
# Collect and process SWTBot code from repositories
python scripts/collect_data.py
```

### 3. Training

```bash
# Start training with default configuration
python scripts/train.py

# Or with custom parameters
python scripts/train.py --max-steps 1000 --batch-size 1 --learning-rate 3e-4
```

### 4. Code Generation

```bash
# Generate code from a prompt
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompt "Create a SWTBot test for clicking a button with text 'Submit'"

# Interactive mode
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --interactive
```

## Environment Setup

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: RTX 3050 Ti (4GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for data and models

### Automatic Setup

The `setup_env.py` script handles most of the setup automatically:

```bash
python scripts/setup_env.py
```

This script will:
- Check Python version compatibility
- Create a virtual environment
- Install all required dependencies
- Create necessary directories
- Check GPU availability
- Generate `.gitignore` file

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw,processed,datasets} models/{checkpoints,final} logs
```

## Data Collection

### Automatic Collection

The data collection script automatically downloads and processes SWTBot code:

```bash
python scripts/collect_data.py
```

### Configuration

Modify `configs/data_config.yaml` to customize data collection:

```yaml
data_sources:
  repositories:
    - url: "https://github.com/eclipse-swtbot/org.eclipse.swtbot"
      name: "swtbot-main"
      branch: "master"
    - url: "https://github.com/caniszczyk/swtbot-examples"
      name: "swtbot-examples"
      branch: "master"
```

### Manual Data Addition

To add your own SWTBot code:

1. Place Java files in `data/raw/custom/files/`
2. Run processing only: `python scripts/collect_data.py --skip-collection`

## Training

### Basic Training

Start training with default settings:

```bash
python scripts/train.py
```

### Custom Training Parameters

```bash
python scripts/train.py \
    --max-steps 2000 \
    --batch-size 2 \
    --learning-rate 5e-4 \
    --output-dir models/my_experiment
```

### Resume Training

```bash
python scripts/train.py --resume-from-checkpoint models/checkpoints/checkpoint-1000
```

### Training with Weights & Biases

```bash
python scripts/train.py --wandb-project my-swtbot-project
```

### Memory Optimization for 4GB GPU

For RTX 3050 Ti and similar GPUs:

```bash
python scripts/train.py \
    --batch-size 1 \
    --max-steps 1500 \
    --learning-rate 3e-4
```

## Code Generation

### Single Prompt Generation

```bash
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompt "Create a SWTBot test for selecting an item from a tree view"
```

### Multiple Samples

Generate multiple variations:

```bash
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompt "Test button click functionality" \
    --num-samples 3 \
    --temperature 0.8
```

### Batch Generation

Create a file `prompts.txt` with one prompt per line:

```
Create a test for login functionality
Test file menu operations
Verify table data display
```

Then run:

```bash
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompts-file prompts.txt \
    --output results.json
```

### Interactive Mode

For continuous generation:

```bash
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --interactive \
    --evaluate
```

### Generation Parameters

Control generation quality:

```bash
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompt "Your prompt here" \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 0.95 \
    --repetition-penalty 1.1
```

## Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  trust_remote_code: true

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 8
  alpha: 32
  dropout: 0.0
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
training:
  max_steps: 2000
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-4
  save_steps: 200
  eval_steps: 200
```

### Data Configuration (`configs/data_config.yaml`)

```yaml
processing:
  min_swtbot_keywords: 2
  min_lines_per_file: 10
  max_lines_per_file: 1000

dataset:
  seq_length: 2048
  fim_rate: 0.5
```

## Example Outputs

### Input Prompt
```
Create a SWTBot test for clicking a button with text 'Submit'
```

### Generated Code
```java
// SWTBot Test Code
// Task: Create a SWTBot test for clicking a button with text 'Submit'

import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class SWTBotTest {
    
    @Test
    public void test() {
        SWTBot bot = new SWTBot();
        
        // Generated test code:
        SWTBotButton submitButton = bot.button("Submit");
        assertTrue("Submit button should be enabled", submitButton.isEnabled());
        submitButton.click();
        
        // Verify the action was performed
        // Add appropriate assertions based on expected behavior
    }
}
```

## Advanced Usage

### Custom Model Training

1. Prepare your own dataset in JSONL format
2. Update data configuration
3. Train with custom parameters:

```bash
python scripts/train.py \
    --data-config configs/custom_data_config.yaml \
    --training-config configs/custom_training_config.yaml
```

### Model Evaluation

Evaluate generated code quality:

```bash
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompts-file test_prompts.txt \
    --evaluate \
    --output evaluation_results.json
```

### Fine-tuning from Checkpoint

Continue training from a specific checkpoint:

```bash
python scripts/train.py \
    --resume-from-checkpoint models/checkpoints/checkpoint-1000 \
    --max-steps 3000
```

## Performance Tips

### For 4GB GPU (RTX 3050 Ti)

1. Use small batch sizes: `--batch-size 1`
2. Increase gradient accumulation: `gradient_accumulation_steps: 16`
3. Enable aggressive quantization
4. Use shorter sequences if needed: `seq_length: 1024`

### For Larger GPUs

1. Increase batch size: `--batch-size 4`
2. Use longer sequences: `seq_length: 4096`
3. Reduce gradient accumulation: `gradient_accumulation_steps: 4`

### Memory Monitoring

Monitor memory usage during training:

```bash
# The training script automatically logs memory usage
# Check logs/training.log for memory statistics
tail -f logs/training.log | grep "memory"
```

## Next Steps

1. **Experiment with Parameters**: Try different learning rates, batch sizes, and LoRA configurations
2. **Add Custom Data**: Include your own SWTBot code for domain-specific training
3. **Evaluate Results**: Use the built-in evaluation tools to assess code quality
4. **Deploy Models**: Use the trained models in your development workflow
5. **Contribute**: Share improvements and findings with the community
