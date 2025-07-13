"""
Data collection module for SWTBot repositories.
Handles cloning repositories and extracting relevant Java files.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    import git
except ImportError:
    print("GitPython not installed. Install with: pip install GitPython")
    git = None
import requests
from tqdm import tqdm
import fnmatch
import json


class DataCollector:
    """Collects data from SWTBot repositories."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "data/raw"):
        """
        Initialize the data collector.
        
        Args:
            config: Data configuration dictionary
            output_dir: Directory to store collected data
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Collection settings
        self.collection_config = config.get("collection", {})
        self.repositories = config.get("data_sources", {}).get("repositories", [])
        
    def collect_all_repositories(self) -> Dict[str, Any]:
        """
        Collect data from all configured repositories.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {
            "repositories_processed": 0,
            "files_collected": 0,
            "total_size_mb": 0,
            "repositories": {}
        }
        
        for repo_config in self.repositories:
            self.logger.info(f"Processing repository: {repo_config['name']}")
            repo_stats = self._collect_repository(repo_config)
            stats["repositories"][repo_config["name"]] = repo_stats
            stats["repositories_processed"] += 1
            stats["files_collected"] += repo_stats["files_collected"]
            stats["total_size_mb"] += repo_stats["size_mb"]
        
        # Save collection metadata
        self._save_collection_metadata(stats)
        
        return stats
    
    def _collect_repository(self, repo_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect data from a single repository.
        
        Args:
            repo_config: Repository configuration
            
        Returns:
            Dictionary with repository statistics
        """
        repo_name = repo_config["name"]
        repo_url = repo_config["url"]
        branch = repo_config.get("branch", "master")
        
        # Create repository directory
        repo_dir = self.output_dir / repo_name
        clone_dir = repo_dir / "clone"
        files_dir = repo_dir / "files"
        
        # Clean up existing directory
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)
        files_dir.mkdir(parents=True)
        
        stats = {
            "url": repo_url,
            "branch": branch,
            "files_collected": 0,
            "size_mb": 0,
            "file_types": {}
        }
        
        try:
            # Clone repository
            self.logger.info(f"Cloning {repo_url}")
            depth = self.collection_config.get("clone_depth", 1)
            repo = git.Repo.clone_from(
                repo_url, 
                clone_dir, 
                branch=branch, 
                depth=depth
            )
            
            # Extract relevant files
            self._extract_files(clone_dir, files_dir, stats)
            
            # Clean up clone directory
            shutil.rmtree(clone_dir)
            
        except Exception as e:
            self.logger.error(f"Error processing repository {repo_name}: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def _extract_files(self, source_dir: Path, target_dir: Path, stats: Dict[str, Any]):
        """
        Extract relevant files from cloned repository.
        
        Args:
            source_dir: Source directory (cloned repo)
            target_dir: Target directory for extracted files
            stats: Statistics dictionary to update
        """
        include_patterns = self.collection_config.get("include_patterns", ["**/*.java"])
        exclude_patterns = self.collection_config.get("exclude_patterns", [])
        max_file_size_mb = self.collection_config.get("max_file_size_mb", 1)
        min_file_size_bytes = self.collection_config.get("min_file_size_bytes", 100)
        
        # Find all files matching include patterns
        all_files = []
        for pattern in include_patterns:
            all_files.extend(source_dir.rglob(pattern))
        
        # Filter files
        valid_files = []
        for file_path in all_files:
            if self._should_include_file(file_path, source_dir, exclude_patterns, 
                                       max_file_size_mb, min_file_size_bytes):
                valid_files.append(file_path)
        
        # Copy valid files
        for file_path in tqdm(valid_files, desc="Extracting files"):
            try:
                # Calculate relative path
                rel_path = file_path.relative_to(source_dir)
                target_path = target_dir / rel_path
                
                # Create target directory
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, target_path)
                
                # Update statistics
                file_size = file_path.stat().st_size
                stats["files_collected"] += 1
                stats["size_mb"] += file_size / (1024 * 1024)
                
                # Track file types
                file_ext = file_path.suffix.lower()
                stats["file_types"][file_ext] = stats["file_types"].get(file_ext, 0) + 1
                
            except Exception as e:
                self.logger.warning(f"Error copying file {file_path}: {e}")
    
    def _should_include_file(self, file_path: Path, base_dir: Path, 
                           exclude_patterns: List[str], max_size_mb: float, 
                           min_size_bytes: int) -> bool:
        """
        Check if a file should be included in the collection.
        
        Args:
            file_path: Path to the file
            base_dir: Base directory for relative path calculation
            exclude_patterns: List of exclude patterns
            max_size_mb: Maximum file size in MB
            min_size_bytes: Minimum file size in bytes
            
        Returns:
            True if file should be included
        """
        try:
            # Check if file exists and is a file
            if not file_path.is_file():
                return False
            
            # Get relative path for pattern matching
            rel_path = file_path.relative_to(base_dir)
            rel_path_str = str(rel_path).replace("\\", "/")  # Normalize path separators
            
            # Check exclude patterns
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(rel_path_str, pattern):
                    return False
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size_mb * 1024 * 1024:
                return False
            if file_size < min_size_bytes:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking file {file_path}: {e}")
            return False
    
    def _save_collection_metadata(self, stats: Dict[str, Any]):
        """
        Save collection metadata to file.
        
        Args:
            stats: Collection statistics
        """
        metadata_file = self.output_dir / "collection_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Collection metadata saved to {metadata_file}")
    
    def get_collected_files(self) -> List[Path]:
        """
        Get list of all collected files.
        
        Returns:
            List of file paths
        """
        files = []
        for repo_dir in self.output_dir.iterdir():
            if repo_dir.is_dir() and (repo_dir / "files").exists():
                files_dir = repo_dir / "files"
                files.extend(files_dir.rglob("*"))
        
        return [f for f in files if f.is_file()]
