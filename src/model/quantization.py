"""
Quantization utilities for memory-efficient training.
Optimized for RTX 3050 Ti (4GB VRAM) constraints.
"""

import logging
import torch
from typing import Dict, Any, Optional
from transformers import BitsAndBytesConfig
import psutil
import gc


def setup_quantization(config: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """
    Setup quantization configuration based on available hardware.
    
    Args:
        config: Quantization configuration dictionary
        
    Returns:
        BitsAndBytesConfig or None if quantization is disabled
    """
    logger = logging.getLogger(__name__)
    
    # Check if quantization is enabled
    if not config.get("load_in_4bit", False) and not config.get("load_in_8bit", False):
        logger.info("Quantization disabled")
        return None
    
    # Auto-detect optimal quantization settings based on available memory
    gpu_memory_gb = get_gpu_memory_gb()
    system_memory_gb = get_system_memory_gb()
    
    logger.info(f"Available GPU memory: {gpu_memory_gb:.1f} GB")
    logger.info(f"Available system memory: {system_memory_gb:.1f} GB")
    
    # Determine optimal quantization settings
    if gpu_memory_gb < 6:  # For RTX 3050 Ti and similar
        logger.info("Low GPU memory detected, using aggressive quantization")
        use_4bit = True
        use_double_quant = True
        compute_dtype = "bfloat16"
    elif gpu_memory_gb < 12:
        logger.info("Medium GPU memory detected, using standard quantization")
        use_4bit = config.get("load_in_4bit", True)
        use_double_quant = config.get("bnb_4bit_use_double_quant", True)
        compute_dtype = config.get("bnb_4bit_compute_dtype", "bfloat16")
    else:
        logger.info("High GPU memory detected, using minimal quantization")
        use_4bit = config.get("load_in_4bit", False)
        use_double_quant = config.get("bnb_4bit_use_double_quant", False)
        compute_dtype = config.get("bnb_4bit_compute_dtype", "float16")
    
    # Convert compute_dtype string to torch dtype
    if isinstance(compute_dtype, str):
        compute_dtype = getattr(torch, compute_dtype)
    
    # Create quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        load_in_8bit=config.get("load_in_8bit", False),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )
    
    logger.info(f"Quantization configured: 4-bit={use_4bit}, double_quant={use_double_quant}")
    return bnb_config


def get_gpu_memory_gb() -> float:
    """
    Get available GPU memory in GB.
    
    Returns:
        GPU memory in GB, or 0 if no GPU available
    """
    try:
        if torch.cuda.is_available():
            # Get memory of the first GPU
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            return gpu_memory_bytes / (1024 ** 3)
        else:
            return 0.0
    except Exception:
        return 0.0


def get_system_memory_gb() -> float:
    """
    Get available system memory in GB.
    
    Returns:
        System memory in GB
    """
    try:
        memory_info = psutil.virtual_memory()
        return memory_info.total / (1024 ** 3)
    except Exception:
        return 16.0  # Default assumption


def optimize_memory_usage():
    """Optimize memory usage by clearing caches and running garbage collection."""
    logger = logging.getLogger(__name__)
    
    # Clear Python garbage
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache")
    
    logger.debug("Memory optimization completed")


def log_memory_usage():
    """Log current memory usage for monitoring."""
    logger = logging.getLogger(__name__)
    
    # System memory
    memory_info = psutil.virtual_memory()
    system_used_gb = (memory_info.total - memory_info.available) / (1024 ** 3)
    system_total_gb = memory_info.total / (1024 ** 3)
    
    logger.info(f"System memory: {system_used_gb:.1f}/{system_total_gb:.1f} GB "
                f"({memory_info.percent:.1f}%)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            
            logger.info(f"GPU {i} memory: {allocated_gb:.1f}GB allocated, "
                       f"{reserved_gb:.1f}GB reserved, {total_gb:.1f}GB total")


def estimate_model_memory(model_name: str, sequence_length: int = 2048, 
                         batch_size: int = 1, use_4bit: bool = True) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        model_name: Name of the model
        sequence_length: Input sequence length
        batch_size: Batch size
        use_4bit: Whether 4-bit quantization is used
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Rough estimates for Llama-3.2-1B-Instruct
    if "1B" in model_name or "1b" in model_name:
        base_params = 1.2e9  # 1.2B parameters
    elif "3B" in model_name or "3b" in model_name:
        base_params = 3.2e9
    elif "7B" in model_name or "7b" in model_name:
        base_params = 7.2e9
    else:
        base_params = 1.2e9  # Default assumption
    
    # Memory calculations
    if use_4bit:
        model_memory_gb = base_params * 0.5 / (1024 ** 3)  # 0.5 bytes per parameter
    else:
        model_memory_gb = base_params * 2 / (1024 ** 3)    # 2 bytes per parameter (fp16)
    
    # Activation memory (rough estimate)
    activation_memory_gb = (batch_size * sequence_length * 4096 * 4) / (1024 ** 3)  # 4 bytes per activation
    
    # Optimizer memory (AdamW needs ~2x model parameters)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Gradient memory
    gradient_memory_gb = model_memory_gb
    
    # Total memory with some overhead
    total_memory_gb = (model_memory_gb + activation_memory_gb + 
                      optimizer_memory_gb + gradient_memory_gb) * 1.2  # 20% overhead
    
    return {
        "model_memory_gb": model_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "gradient_memory_gb": gradient_memory_gb,
        "total_memory_gb": total_memory_gb,
        "recommended_gpu_memory_gb": total_memory_gb * 1.5  # Safety margin
    }


def check_memory_requirements(model_name: str, sequence_length: int = 2048,
                            batch_size: int = 1, use_4bit: bool = True) -> bool:
    """
    Check if current hardware can handle the model requirements.
    
    Args:
        model_name: Name of the model
        sequence_length: Input sequence length
        batch_size: Batch size
        use_4bit: Whether 4-bit quantization is used
        
    Returns:
        True if requirements can be met
    """
    logger = logging.getLogger(__name__)
    
    # Get memory estimates
    estimates = estimate_model_memory(model_name, sequence_length, batch_size, use_4bit)
    
    # Get available memory
    gpu_memory_gb = get_gpu_memory_gb()
    system_memory_gb = get_system_memory_gb()
    
    # Check GPU memory
    gpu_sufficient = gpu_memory_gb >= estimates["total_memory_gb"]
    
    # Check system memory (for offloading if needed)
    system_sufficient = system_memory_gb >= estimates["total_memory_gb"] * 2
    
    logger.info(f"Memory requirements check:")
    logger.info(f"  Estimated total memory needed: {estimates['total_memory_gb']:.1f} GB")
    logger.info(f"  Available GPU memory: {gpu_memory_gb:.1f} GB")
    logger.info(f"  Available system memory: {system_memory_gb:.1f} GB")
    logger.info(f"  GPU sufficient: {gpu_sufficient}")
    logger.info(f"  System sufficient: {system_sufficient}")
    
    if not gpu_sufficient and not system_sufficient:
        logger.warning("Insufficient memory for training. Consider:")
        logger.warning("  - Reducing batch size")
        logger.warning("  - Reducing sequence length")
        logger.warning("  - Using more aggressive quantization")
        logger.warning("  - Using gradient accumulation")
        return False
    
    return True
