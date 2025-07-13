# SWTBot Code Fine-tuning Pipeline

A modular fine-tuning pipeline for training Llama-3.2-1B-Instruct to generate SWTBot test code in Java.

## Features

- **Memory Efficient**: Optimized for RTX 3050 Ti (4GB VRAM) using 4-bit quantization and LoRA
- **Modular Design**: Easy to maintain and extend
- **Checkpoint Saving**: Saves model after configurable training epochs
- **SWTBot Focused**: Specialized for Eclipse SWTBot test automation code generation
- **Virtual Environment**: Isolated Python environment for dependencies

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py          # Data collection from repositories
│   │   ├── processor.py          # Java code processing and filtering
│   │   └── dataset.py            # Dataset preparation and tokenization
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py             # Model configuration and setup
│   │   └── quantization.py       # Quantization utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training pipeline
│   │   ├── callbacks.py          # Custom training callbacks
│   │   └── utils.py              # Training utilities
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generator.py          # Code generation pipeline
│   │   └── evaluator.py          # Model evaluation utilities
│   └── utils/
│       ├── __init__.py
│       ├── logging.py            # Logging configuration
│       └── config.py             # Global configuration
├── configs/
│   ├── model_config.yaml         # Model configuration
│   ├── training_config.yaml      # Training parameters
│   └── data_config.yaml          # Data processing configuration
├── scripts/
│   ├── setup_env.py              # Environment setup
│   ├── collect_data.py           # Data collection script
│   ├── train.py                  # Main training script
│   └── generate.py               # Code generation script
├── data/
│   ├── raw/                      # Raw collected data
│   ├── processed/                # Processed training data
│   └── datasets/                 # Final datasets
├── models/
│   ├── checkpoints/              # Training checkpoints
│   └── final/                    # Final trained models
├── logs/                         # Training logs
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Quick Start

1. **Setup Environment**:
   ```bash
   python scripts/setup_env.py
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Create Sample Data** (for quick testing):
   ```bash
   python create_sample_data.py
   ```

3. **Train Model** (minimal version for compatibility):
   ```bash
   python train_minimal.py --batch-size 1 --max-steps 500
   ```

4. **Generate Code**:
   ```bash
   python scripts/generate.py --model-path ./minimal_model --prompt "Create a SWTBot test for button click"
   ```

### Alternative: Full Pipeline

For the complete data collection and training pipeline:

```bash
# Collect real data from repositories
python scripts/collect_data.py

# Train with full pipeline
python scripts/train.py --batch-size 1 --max-steps 500
```

## Configuration

All configurations are stored in YAML files in the `configs/` directory:

- `model_config.yaml`: Model architecture and quantization settings
- `training_config.yaml`: Training hyperparameters and checkpoint settings
- `data_config.yaml`: Data collection and processing parameters

## Hardware Requirements

- **GPU**: RTX 3050 Ti (4GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for data and models

## Data Sources

- [Eclipse SWTBot](https://github.com/eclipse-swtbot/org.eclipse.swtbot)
- [SWTBot Examples](https://github.com/caniszczyk/swtbot-examples)

## Documentation

- **[Usage Guide](docs/USAGE.md)**: Comprehensive guide for using the pipeline
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Solutions for common issues
- **[Configuration Examples](examples/config_examples.md)**: Hardware-specific configurations
- **[Sample Prompts](examples/sample_prompts.txt)**: Example prompts for code generation

## Example Usage

### Training a Model

```bash
# Setup environment
python scripts/setup_env.py
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create sample data for quick start
python create_sample_data.py

# Start training (minimal version for compatibility)
python train_minimal.py --max-steps 500 --batch-size 1

# Or use full pipeline with real data
python scripts/collect_data.py
python scripts/train.py --max-steps 500 --batch-size 1
```

### Generating SWTBot Code

```bash
# Generate code from a prompt (using minimal trained model)
python scripts/generate.py \
    --model-path ./minimal_model \
    --prompt "Create a SWTBot test for clicking a button with text 'Submit'" \
    --evaluate

# Interactive generation
python scripts/generate.py \
    --model-path ./minimal_model \
    --interactive

# Using full pipeline model
python scripts/generate.py \
    --model-path models/checkpoints/final_model \
    --prompt "Your prompt here"
```

### Example Output

**Input Prompt:**
```
Create a SWTBot test for clicking a button with text 'Submit'
```

**Generated Code:**
```java
import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class SWTBotTest {

    @Test
    public void testSubmitButton() {
        SWTBot bot = new SWTBot();

        // Find and click the Submit button
        SWTBotButton submitButton = bot.button("Submit");
        assertTrue("Submit button should be enabled", submitButton.isEnabled());
        submitButton.click();

        // Verify the action was performed
        // Add appropriate assertions based on expected behavior
    }
}
```

## Performance Benchmarks

| GPU Model | VRAM | Batch Size | Training Speed | Memory Usage |
|-----------|------|------------|----------------|--------------|
| RTX 3050 Ti | 4GB | 1 | ~0.5 steps/sec | ~3.5GB |
| RTX 4060 | 8GB | 2 | ~1.0 steps/sec | ~6GB |
| RTX 4070 | 12GB | 4 | ~1.5 steps/sec | ~9GB |

## Advanced Features

- **Memory Optimization**: 4-bit quantization with LoRA for 4GB GPUs
- **Checkpoint Saving**: Automatic model saving every N training steps
- **Code Evaluation**: Built-in quality assessment for generated code
- **Interactive Generation**: Real-time code generation with feedback
- **Batch Processing**: Generate code for multiple prompts simultaneously
- **Weights & Biases Integration**: Comprehensive training monitoring

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library and model hosting
- [Eclipse SWTBot](https://www.eclipse.org/swtbot/) team for the testing framework
- [Meta](https://ai.meta.com/) for the Llama model family
- [Microsoft](https://github.com/microsoft/LoRA) for the LoRA technique

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{swtbot_finetune_2024,
  title={SWTBot Code Fine-tuning Pipeline},
  author={SWTBot Fine-tuning Team},
  year={2024},
  url={https://github.com/your-repo/swtbot-finetune}
}
```