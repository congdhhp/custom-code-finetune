"""
SWTBot code generation module.
Handles loading trained models and generating SWTBot test code.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
try:
    from peft import PeftModel
except ImportError:
    print("PEFT not installed. Install with: pip install peft")
    PeftModel = None
import re


class SWTBotCodeGenerator:
    """Generator for SWTBot test code using fine-tuned models."""
    
    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SWTBot code generator.
        
        Args:
            model_path: Path to the trained model
            config: Optional configuration dictionary
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Generation settings
        self.default_generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": None,  # Will be set from tokenizer
        }
        
        # Load model and tokenizer
        self._load_model()
        self._setup_generation_config()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        # Check if it's a PEFT model or full model
        if (self.model_path / "adapter_config.json").exists():
            self._load_peft_model()
        else:
            self._load_full_model()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info("Model and tokenizer loaded successfully")
    
    def _load_peft_model(self):
        """Load PEFT (LoRA) model."""
        # Load base model first
        base_model_name = self._get_base_model_name()
        
        # Setup quantization if needed
        quantization_config = self._get_quantization_config()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Load PEFT model
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.merge_and_unload()  # Merge LoRA weights
        
        self.logger.info("PEFT model loaded and merged")
    
    def _load_full_model(self):
        """Load full fine-tuned model."""
        quantization_config = self._get_quantization_config()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        self.logger.info("Full model loaded")
    
    def _get_base_model_name(self) -> str:
        """Get base model name from PEFT config."""
        import json
        
        config_file = self.model_path / "adapter_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                peft_config = json.load(f)
                return peft_config.get("base_model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct")
        
        return "meta-llama/Llama-3.2-1B-Instruct"
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for inference."""
        if not self.config.get("use_quantization", True):
            return None
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    def _setup_generation_config(self):
        """Setup generation configuration."""
        generation_config = self.default_generation_config.copy()
        generation_config.update(self.config.get("generation", {}))
        
        # Set pad token ID
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        
        self.generation_config = GenerationConfig(**generation_config)
        self.logger.info("Generation configuration setup complete")
    
    def generate_code(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate SWTBot code from a prompt.
        
        Args:
            prompt: Input prompt describing the desired test
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated SWTBot code
        """
        # Prepare the prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            # Merge generation config with kwargs
            gen_config = self.generation_config
            if generation_kwargs:
                gen_config = GenerationConfig(**{
                    **self.generation_config.to_dict(),
                    **generation_kwargs
                })
            
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                do_sample=gen_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Post-process the generated code
        cleaned_code = self._post_process_code(generated_text)
        
        return cleaned_code
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format the input prompt for code generation.
        
        Args:
            prompt: Raw input prompt
            
        Returns:
            Formatted prompt
        """
        # Add SWTBot context if not present
        if "SWTBot" not in prompt and "swtbot" not in prompt.lower():
            prompt = f"Create a SWTBot test for: {prompt}"
        
        # Add Java context
        formatted_prompt = f"""// SWTBot Test Code
// Task: {prompt}

import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class SWTBotTest {{
    
    @Test
    public void test() {{
        SWTBot bot = new SWTBot();
        
        // Generated test code:
"""
        
        return formatted_prompt
    
    def _post_process_code(self, generated_code: str) -> str:
        """
        Post-process generated code to clean it up.
        
        Args:
            generated_code: Raw generated code
            
        Returns:
            Cleaned code
        """
        # Remove incomplete lines at the end
        lines = generated_code.split('\n')
        
        # Find the last complete statement
        cleaned_lines = []
        for line in lines:
            cleaned_lines.append(line)
            # Stop at closing brace or if we hit incomplete code
            if line.strip() == '}' and len(cleaned_lines) > 5:
                break
        
        # Join lines and clean up
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Remove any trailing incomplete code
        cleaned_code = re.sub(r'\n\s*$', '', cleaned_code)
        
        # Ensure proper indentation
        cleaned_code = self._fix_indentation(cleaned_code)
        
        return cleaned_code
    
    def _fix_indentation(self, code: str) -> str:
        """Fix indentation in generated code."""
        lines = code.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Decrease indent for closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Add proper indentation
            fixed_lines.append('    ' * indent_level + stripped)
            
            # Increase indent for opening braces
            if stripped.endswith('{'):
                indent_level += 1
        
        return '\n'.join(fixed_lines)
    
    def generate_multiple(self, prompt: str, num_samples: int = 3, 
                         **generation_kwargs) -> List[str]:
        """
        Generate multiple code samples for the same prompt.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated code samples
        """
        samples = []
        
        # Ensure sampling is enabled
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.8)
        
        for i in range(num_samples):
            self.logger.debug(f"Generating sample {i+1}/{num_samples}")
            code = self.generate_code(prompt, **generation_kwargs)
            samples.append(code)
        
        return samples
    
    def generate_with_context(self, prompt: str, context: str = "", 
                            **generation_kwargs) -> str:
        """
        Generate code with additional context.
        
        Args:
            prompt: Main prompt
            context: Additional context (existing code, requirements, etc.)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated code
        """
        if context:
            full_prompt = f"{context}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        return self.generate_code(full_prompt, **generation_kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        info = {
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "tokenizer_type": type(self.tokenizer).__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": getattr(self.tokenizer, 'model_max_length', 'unknown'),
        }
        
        # Add parameter count if available
        if hasattr(self.model, 'num_parameters'):
            info["num_parameters"] = self.model.num_parameters()
        
        return info
