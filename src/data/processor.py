"""
Java code processing module for SWTBot fine-tuning.
Handles filtering, cleaning, and preparing Java code for training.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from tqdm import tqdm
import hashlib


class JavaCodeProcessor:
    """Processes Java code files for SWTBot fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Java code processor.
        
        Args:
            config: Processing configuration dictionary
        """
        self.config = config
        self.processing_config = config.get("processing", {})
        self.quality_config = config.get("dataset", {}).get("quality_control", {})
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # SWTBot keywords for filtering
        self.swtbot_keywords = self.processing_config.get("swtbot_keywords", [])
        
        # Compiled regex patterns for efficiency
        self._compile_patterns()
        
        # Statistics tracking
        self.stats = {
            "files_processed": 0,
            "files_accepted": 0,
            "files_rejected": 0,
            "rejection_reasons": {},
            "total_lines": 0,
            "total_characters": 0
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for code processing."""
        # Pattern for removing excessive whitespace
        self.whitespace_pattern = re.compile(r'\n\s*\n\s*\n', re.MULTILINE)
        
        # Pattern for Java comments
        self.single_line_comment = re.compile(r'//.*$', re.MULTILINE)
        self.multi_line_comment = re.compile(r'/\*.*?\*/', re.DOTALL)
        
        # Pattern for import statements
        self.import_pattern = re.compile(r'^import\s+[^;]+;', re.MULTILINE)
        
        # Pattern for package declaration
        self.package_pattern = re.compile(r'^package\s+[^;]+;', re.MULTILINE)
        
        # Pattern for class/interface declaration
        self.class_pattern = re.compile(r'(public\s+)?(abstract\s+)?(class|interface|enum)\s+\w+')
        
        # Pattern for method declaration
        self.method_pattern = re.compile(r'(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{')
    
    def process_files(self, input_files: List[Path], output_dir: str) -> Dict[str, Any]:
        """
        Process a list of Java files.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory for processed files
            
        Returns:
            Processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_files = []
        
        for file_path in tqdm(input_files, desc="Processing Java files"):
            if file_path.suffix.lower() == '.java':
                result = self._process_single_file(file_path)
                if result:
                    processed_files.append(result)
        
        # Save processed files
        self._save_processed_files(processed_files, output_path)
        
        # Update final statistics
        self.stats["files_accepted"] = len(processed_files)
        
        return self.stats
    
    def _process_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single Java file.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            Processed file data or None if rejected
        """
        self.stats["files_processed"] += 1
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic validation
            if not self._validate_file_content(content, file_path):
                return None
            
            # Clean and normalize content
            processed_content = self._clean_content(content)
            
            # Extract metadata
            metadata = self._extract_metadata(processed_content, file_path)
            
            # Final validation
            if not self._final_validation(processed_content, metadata):
                return None
            
            # Update statistics
            self.stats["total_lines"] += len(processed_content.split('\n'))
            self.stats["total_characters"] += len(processed_content)
            
            return {
                "file_path": str(file_path),
                "content": processed_content,
                "metadata": metadata,
                "hash": hashlib.md5(processed_content.encode()).hexdigest()
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing file {file_path}: {e}")
            self._update_rejection_stats("processing_error")
            return None
    
    def _validate_file_content(self, content: str, file_path: Path) -> bool:
        """
        Validate file content against basic criteria.
        
        Args:
            content: File content
            file_path: File path for logging
            
        Returns:
            True if content is valid
        """
        lines = content.split('\n')
        
        # Check line count
        min_lines = self.processing_config.get("min_lines_per_file", 10)
        max_lines = self.processing_config.get("max_lines_per_file", 1000)
        
        if len(lines) < min_lines:
            self._update_rejection_stats("too_few_lines")
            return False
        
        if len(lines) > max_lines:
            self._update_rejection_stats("too_many_lines")
            return False
        
        # Check for SWTBot keywords
        min_keywords = self.processing_config.get("min_swtbot_keywords", 2)
        keyword_count = sum(1 for keyword in self.swtbot_keywords if keyword in content)
        
        if keyword_count < min_keywords:
            self._update_rejection_stats("insufficient_swtbot_keywords")
            return False
        
        # Check if it's a valid Java file
        if not (self.class_pattern.search(content) or self.method_pattern.search(content)):
            self._update_rejection_stats("no_java_structure")
            return False
        
        return True

    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize Java code content.

        Args:
            content: Raw file content

        Returns:
            Cleaned content
        """
        # Remove or preserve comments based on configuration
        if not self.processing_config.get("include_comments", True):
            content = self.single_line_comment.sub('', content)

        if not self.processing_config.get("include_javadoc", True):
            content = self.multi_line_comment.sub('', content)

        # Normalize whitespace
        if self.processing_config.get("normalize_whitespace", True):
            content = self.whitespace_pattern.sub('\n\n', content)
            content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces and tabs

        # Remove excessive empty lines but preserve some structure
        if not self.processing_config.get("remove_empty_lines", False):
            content = re.sub(r'\n{3,}', '\n\n', content)

        # Preserve indentation if requested
        if self.processing_config.get("preserve_indentation", True):
            # Keep original indentation structure
            pass
        else:
            # Normalize indentation to 4 spaces
            lines = content.split('\n')
            normalized_lines = []
            for line in lines:
                if line.strip():
                    # Calculate indentation level
                    indent_level = len(line) - len(line.lstrip())
                    normalized_indent = '    ' * (indent_level // 4)
                    normalized_lines.append(normalized_indent + line.lstrip())
                else:
                    normalized_lines.append('')
            content = '\n'.join(normalized_lines)

        return content.strip()

    def _extract_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from Java code.

        Args:
            content: Processed content
            file_path: Original file path

        Returns:
            Metadata dictionary
        """
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "line_count": len(content.split('\n')),
            "char_count": len(content),
            "swtbot_keywords": [],
            "imports": [],
            "package": None,
            "classes": [],
            "methods": []
        }

        # Find SWTBot keywords
        for keyword in self.swtbot_keywords:
            if keyword in content:
                metadata["swtbot_keywords"].append(keyword)

        # Extract imports
        imports = self.import_pattern.findall(content)
        metadata["imports"] = [imp.strip() for imp in imports]

        # Extract package declaration
        package_match = self.package_pattern.search(content)
        if package_match:
            metadata["package"] = package_match.group(0).strip()

        # Extract class declarations
        class_matches = self.class_pattern.findall(content)
        metadata["classes"] = [match[-1] if isinstance(match, tuple) else match for match in class_matches]

        # Extract method declarations (simplified)
        method_matches = self.method_pattern.findall(content)
        metadata["methods"] = len(method_matches)

        return metadata

    def _final_validation(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Perform final validation on processed content.

        Args:
            content: Processed content
            metadata: Extracted metadata

        Returns:
            True if content passes final validation
        """
        # Check for minimum unique tokens
        min_unique_tokens = self.quality_config.get("min_unique_tokens", 50)
        unique_tokens = set(re.findall(r'\w+', content.lower()))
        if len(unique_tokens) < min_unique_tokens:
            self._update_rejection_stats("insufficient_unique_tokens")
            return False

        # Check for excessive repetition
        max_repetition_ratio = self.quality_config.get("max_repetition_ratio", 0.3)
        total_tokens = len(re.findall(r'\w+', content))
        if total_tokens > 0:
            repetition_ratio = 1 - (len(unique_tokens) / total_tokens)
            if repetition_ratio > max_repetition_ratio:
                self._update_rejection_stats("excessive_repetition")
                return False

        return True

    def _update_rejection_stats(self, reason: str):
        """Update rejection statistics."""
        self.stats["files_rejected"] += 1
        self.stats["rejection_reasons"][reason] = self.stats["rejection_reasons"].get(reason, 0) + 1

    def _save_processed_files(self, processed_files: List[Dict[str, Any]], output_dir: Path):
        """
        Save processed files to output directory.

        Args:
            processed_files: List of processed file data
            output_dir: Output directory
        """
        # Save as JSONL format
        output_file = output_dir / "processed_java_files.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for file_data in processed_files:
                f.write(json.dumps(file_data, ensure_ascii=False) + '\n')

        # Save statistics
        stats_file = output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

        self.logger.info(f"Processed {len(processed_files)} files saved to {output_file}")
        self.logger.info(f"Processing statistics saved to {stats_file}")

    def remove_duplicates(self, processed_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate files based on content hash.

        Args:
            processed_files: List of processed file data

        Returns:
            List with duplicates removed
        """
        if not self.quality_config.get("remove_duplicates", True):
            return processed_files

        seen_hashes = set()
        unique_files = []

        for file_data in processed_files:
            file_hash = file_data.get("hash")
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files.append(file_data)

        removed_count = len(processed_files) - len(unique_files)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate files")

        return unique_files
