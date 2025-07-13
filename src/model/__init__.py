"""Model configuration and setup modules."""

from .config import ModelConfig
from .quantization import setup_quantization

__all__ = ["ModelConfig", "setup_quantization"]
