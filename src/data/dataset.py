"""
Dataset preparation module for SWTBot fine-tuning.
Handles tokenization, FIM transformations, and constant-length chunking.
"""

import json
import logging
import functools
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class SWTBotDataset(Dataset):
    """Dataset class for SWTBot Java code files."""
    
    def __init__(self, data_file: str, tokenizer_name: str, max_length: int = 2048):
        """
        Initialize the SWTBot dataset.
        
        Args:
            data_file: Path to processed JSONL file
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
        """
        self.data_file = Path(data_file)
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.data = self._load_data()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.data)} examples from {data_file}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        example = self.data[idx]
        content = example["content"]
        
        # Tokenize
        encoding = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For causal LM
        }


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    Adapted from the HuggingFace cookbook example for SWTBot Java code.
    """
    
    def __init__(
        self,
        tokenizer,
        dataset,
        infinite: bool = False,
        seq_length: int = 2048,
        num_of_sequences: int = 1024,
        chars_per_token: float = 2.5,
        content_field: str = "content",
        fim_rate: float = 0.5,
        fim_spm_rate: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize the constant length dataset.
        
        Args:
            tokenizer: Tokenizer to use
            dataset: Source dataset (iterable)
            infinite: If True, reset iterator when dataset ends
            seq_length: Length of token sequences to return
            num_of_sequences: Number of token sequences to keep in buffer
            chars_per_token: Number of characters per token (for buffer sizing)
            content_field: Field name containing the code content
            fim_rate: Rate (0.0 to 1.0) that sample will be permuted with FIM
            fim_spm_rate: Rate (0.0 to 1.0) of FIM permutations that will use SPM
            seed: Seed for random number generator
        """
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed
        
        # Get FIM token IDs
        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = self._get_fim_token_ids()
        
        if not self.suffix_tok_id and self.fim_rate > 0:
            logging.warning("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0
    
    @functools.lru_cache(maxsize=None)
    def _get_fim_token_ids(self):
        """Get token IDs for FIM special tokens."""
        try:
            special_tokens = self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
            if len(special_tokens) >= 5:
                FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = special_tokens[1:5]
                suffix_tok_id = self.tokenizer.vocab.get(FIM_SUFFIX)
                prefix_tok_id = self.tokenizer.vocab.get(FIM_PREFIX)
                middle_tok_id = self.tokenizer.vocab.get(FIM_MIDDLE)
                pad_tok_id = self.tokenizer.vocab.get(FIM_PAD)
                return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id
        except (KeyError, AttributeError):
            pass
        return None, None, None, None
    
    def _permute_fim(
        self,
        sample: List[int],
        np_rng: np.random.RandomState,
        fim_rate: float = 0.5,
        fim_spm_rate: float = 0.5,
        truncate_or_pad: bool = False,
    ) -> List[int]:
        """
        Apply FIM (Fill-in-the-Middle) transformation to a sample.
        
        Args:
            sample: List of token IDs
            np_rng: Random number generator
            fim_rate: Probability of applying FIM transformation
            fim_spm_rate: Probability of using SPM variant
            truncate_or_pad: Whether to truncate or pad to maintain length
            
        Returns:
            Transformed sample
        """
        if not self.suffix_tok_id or not np_rng.binomial(1, fim_rate):
            return sample
        
        # Split the sample into prefix, middle, and suffix
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()
        
        prefix = np.array(sample[:boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0]:boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1]:], dtype=np.int64)
        
        if truncate_or_pad:
            # Calculate new length with FIM tokens
            new_length = len(suffix) + len(prefix) + len(middle) + 3
            diff = new_length - len(sample)
            
            if diff > 0:
                if len(suffix) <= diff:
                    return sample
                suffix = suffix[:len(suffix) - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), self.pad_tok_id)])
        
        # Apply SPM (Suffix-Prefix-Middle) or PSM (Prefix-Suffix-Middle)
        if np_rng.binomial(1, fim_spm_rate):
            # SPM variant
            new_sample = np.concatenate([
                [self.prefix_tok_id, self.suffix_tok_id],
                suffix,
                [self.middle_tok_id],
                prefix,
                middle,
            ])
        else:
            # PSM variant
            new_sample = np.concatenate([
                [self.prefix_tok_id],
                prefix,
                [self.suffix_tok_id],
                suffix,
                [self.middle_tok_id],
                middle,
            ])
        
        return list(new_sample)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset."""
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        
        while more_examples:
            buffer, buffer_len = [], 0
            
            # Fill buffer
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    example = next(iterator)
                    if isinstance(example, dict):
                        content = example.get(self.content_field, "")
                    else:
                        content = str(example)
                    
                    buffer.append(content)
                    buffer_len += len(content)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            
            if not buffer:
                break
            
            # Tokenize buffer
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            
            for tokenized_input in tokenized_inputs:
                # Apply FIM transformations
                if self.fim_rate > 0:
                    tokenized_input = self._permute_fim(
                        tokenized_input,
                        np_rng,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )
                
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            
            # Create examples of constant length
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i:i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            
            # Shuffle examples
            random.shuffle(examples)
            
            # Yield examples
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
