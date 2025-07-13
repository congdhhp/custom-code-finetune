# Getting Started - SWTBot Fine-tuning Pipeline

This guide helps you get started quickly with the SWTBot fine-tuning pipeline.

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- RTX 3050 Ti (4GB VRAM) or better

## Quick Setup

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers peft bitsandbytes accelerate datasets pyyaml tqdm rich
```

### 2. Setup Environment (Optional but Recommended)

```bash
python scripts/setup_env.py
```

This will create a virtual environment and install all dependencies automatically.

## Training Options

### Option 1: Quick Start with Sample Data (Recommended)

```bash
# 1. Create sample training data
python create_sample_data.py

# 2. Train with minimal script (most compatible)
python train_minimal.py --batch-size 1 --max-steps 500

# 3. Generate code
python scripts/generate.py --model-path ./minimal_model --prompt "Create a SWTBot test for button click"
```

### Option 2: Full Pipeline with Real Data

```bash
# 1. Collect real SWTBot code from repositories
python scripts/collect_data.py

# 2. Train with full pipeline
python scripts/train.py --batch-size 1 --max-steps 500

# 3. Generate code
python scripts/generate.py --model-path models/checkpoints/final_model --prompt "Your prompt"
```

## Key Files

- **`train_minimal.py`**: Simplified, compatible training script
- **`create_sample_data.py`**: Creates sample SWTBot training data
- **`scripts/train.py`**: Full training pipeline
- **`scripts/generate.py`**: Code generation script
- **`configs/`**: Configuration files for model, training, and data

## Hardware Optimization

### For RTX 3050 Ti (4GB VRAM):
- Use `--batch-size 1`
- Max steps: 500-1000 for testing
- The pipeline automatically uses 4-bit quantization and LoRA

### For Larger GPUs:
- Increase `--batch-size` to 2-4
- Increase `--max-steps` to 2000+
- Modify configs for better performance

## Troubleshooting

### Common Issues:

1. **"No module named 'torch'"**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **CUDA out of memory**
   ```bash
   python train_minimal.py --batch-size 1 --max-length 256
   ```

3. **Import errors**
   ```bash
   pip install transformers peft bitsandbytes
   ```

4. **Training fails**
   - Use `train_minimal.py` instead of `scripts/train.py`
   - Reduce `--max-steps` and `--batch-size`

## Expected Results

After training for 500 steps, you should get a model that can generate SWTBot test code like:

```java
import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotButton;
import org.junit.Test;

public class ButtonTest {
    @Test
    public void testButtonClick() {
        SWTBot bot = new SWTBot();
        SWTBotButton button = bot.button("Submit");
        button.click();
    }
}
```

## Next Steps

1. **Experiment with prompts**: Try different test scenarios
2. **Increase training**: Use more steps for better quality
3. **Add custom data**: Include your own SWTBot code
4. **Evaluate results**: Use the built-in evaluation features

For detailed documentation, see:
- `docs/USAGE.md` - Complete usage guide
- `docs/TROUBLESHOOTING.md` - Detailed troubleshooting
- `examples/` - Configuration examples and sample prompts
