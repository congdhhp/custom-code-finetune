"""
Model configuration module for SWTBot fine-tuning.
Handles model setup, LoRA configuration, and memory optimization.
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


class ModelConfig:
    """Handles model configuration and setup for SWTBot fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.quantization_config = config.get("quantization", {})
        self.lora_config = config.get("lora", {})
        self.tokenizer_config = config.get("tokenizer", {})
        self.memory_config = config.get("memory", {})
        
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """
        Setup and configure the tokenizer.
        
        Returns:
            Configured tokenizer
        """
        model_name = self.model_config.get("name", "meta-llama/Llama-3.2-1B-Instruct")
        trust_remote_code = self.model_config.get("trust_remote_code", True)
        
        self.logger.info(f"Loading tokenizer: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side=self.tokenizer_config.get("padding_side", "right")
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.logger.info("Set pad_token to eos_token")
        
        # Configure tokenizer settings
        max_length = self.tokenizer_config.get("max_length", 2048)
        self.tokenizer.model_max_length = max_length
        
        self.logger.info(f"Tokenizer configured with max_length: {max_length}")
        return self.tokenizer
    
    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Setup quantization configuration for memory efficiency.
        
        Returns:
            BitsAndBytesConfig or None if quantization is disabled
        """
        if not self.quantization_config.get("load_in_4bit", False):
            self.logger.info("Quantization disabled")
            return None
        
        compute_dtype = self.quantization_config.get("bnb_4bit_compute_dtype", "bfloat16")
        if isinstance(compute_dtype, str):
            compute_dtype = getattr(torch, compute_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quantization_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=self.quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.quantization_config.get("bnb_4bit_use_double_quant", True),
            load_in_8bit=self.quantization_config.get("load_in_8bit", False),
        )
        
        self.logger.info("4-bit quantization configured")
        return bnb_config
    
    def setup_model(self, quantization_config: Optional[BitsAndBytesConfig] = None) -> AutoModelForCausalLM:
        """
        Setup and configure the base model.
        
        Args:
            quantization_config: Optional quantization configuration
            
        Returns:
            Configured model
        """
        model_name = self.model_config.get("name", "meta-llama/Llama-3.2-1B-Instruct")
        trust_remote_code = self.model_config.get("trust_remote_code", True)
        use_cache = self.model_config.get("use_cache", False)
        use_flash_attention_2 = self.model_config.get("use_flash_attention_2", True)
        device_map = self.memory_config.get("device_map", "auto")
        
        self.logger.info(f"Loading model: {model_name}")
        
        # Model loading arguments
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "use_cache": use_cache,
            "device_map": device_map,
        }
        
        # Add quantization config if provided
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Add flash attention if supported (try to load model first to check compatibility)
        if use_flash_attention_2:
            model_kwargs["use_flash_attention_2"] = True
            self.logger.info("Flash Attention 2 requested")
        
        # Load model with error handling for unsupported features
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if use_flash_attention_2:
                self.logger.info("Flash Attention 2 enabled successfully")
        except TypeError as e:
            if "use_flash_attention_2" in str(e):
                self.logger.warning("Flash Attention 2 not supported by this model, disabling...")
                model_kwargs.pop("use_flash_attention_2", None)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                raise e
        
        # Prepare model for k-bit training if quantized
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
            self.logger.info("Model prepared for k-bit training")
        
        # Enable gradient checkpointing for memory efficiency
        if self.memory_config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        return self.model
    
    def setup_lora_config(self) -> LoraConfig:
        """
        Setup LoRA configuration for parameter-efficient fine-tuning.
        
        Returns:
            LoRA configuration
        """
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 8),
            lora_alpha=self.lora_config.get("alpha", 32),
            lora_dropout=self.lora_config.get("dropout", 0.0),
            bias=self.lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
        )
        
        self.logger.info(f"LoRA configured with r={lora_config.r}, alpha={lora_config.lora_alpha}")
        return lora_config
    
    def setup_peft_model(self, model: AutoModelForCausalLM, lora_config: LoraConfig):
        """
        Setup PEFT model with LoRA.
        
        Args:
            model: Base model
            lora_config: LoRA configuration
            
        Returns:
            PEFT model
        """
        self.peft_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params, all_params = self.peft_model.get_nb_trainable_parameters()
        trainable_percentage = 100 * trainable_params / all_params
        
        self.logger.info(
            f"Trainable parameters: {trainable_params:,} / {all_params:,} "
            f"({trainable_percentage:.2f}%)"
        )
        
        return self.peft_model
    
    def setup_complete_model(self) -> tuple:
        """
        Setup complete model pipeline with tokenizer, quantization, and LoRA.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info("Setting up complete model pipeline")
        
        # Setup tokenizer
        tokenizer = self.setup_tokenizer()
        
        # Setup quantization
        quantization_config = self.setup_quantization_config()
        
        # Setup base model
        model = self.setup_model(quantization_config)
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        peft_model = self.setup_peft_model(model, lora_config)
        
        self.logger.info("Model pipeline setup complete")
        return peft_model, tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Model information dictionary
        """
        if self.peft_model is None:
            return {"error": "Model not configured"}
        
        trainable_params, all_params = self.peft_model.get_nb_trainable_parameters()
        
        info = {
            "model_name": self.model_config.get("name"),
            "total_parameters": all_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / all_params,
            "quantization_enabled": self.quantization_config.get("load_in_4bit", False),
            "lora_r": self.lora_config.get("r", 8),
            "lora_alpha": self.lora_config.get("alpha", 32),
            "max_sequence_length": self.tokenizer_config.get("max_length", 2048),
        }
        
        return info
    
    def save_model(self, output_dir: str):
        """
        Save the trained model.
        
        Args:
            output_dir: Output directory for saving the model
        """
        if self.peft_model is None:
            raise ValueError("Model not configured")
        
        self.peft_model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Model saved to {output_dir}")
