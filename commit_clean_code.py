#!/usr/bin/env python3
"""
Script to commit the cleaned up SWTBot fine-tuning pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, check=True):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def main():
    """Main commit function."""
    print("Committing cleaned SWTBot fine-tuning pipeline...")
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("Initializing git repository...")
        run_command("git init")
    
    # Add all files
    print("Adding files to git...")
    run_command("git add .")
    
    # Check git status
    print("Git status:")
    run_command("git status")
    
    # Commit with a descriptive message
    commit_message = """Clean SWTBot Fine-tuning Pipeline

Features:
- ✅ RTX 3050 Ti (4GB VRAM) optimized training
- ✅ Checkpoint saving during training
- ✅ Minimal training script for compatibility
- ✅ Sample data generation for quick start
- ✅ Full pipeline with real data collection
- ✅ Code generation and evaluation
- ✅ Comprehensive documentation

Key Files:
- train_minimal.py: Compatible training script
- create_sample_data.py: Sample data generation
- scripts/: Full pipeline scripts
- configs/: Configuration files
- docs/: Detailed documentation

Hardware Support:
- 4-bit quantization with LoRA
- Memory optimization for low VRAM
- CUDA 11.8+ support
- Automatic GPU detection

Ready for production use!"""
    
    print("Committing changes...")
    run_command(f'git commit -m "{commit_message}"')
    
    print("\n" + "="*60)
    print("✅ Code committed successfully!")
    print("="*60)
    print("\nRepository is now clean and ready for use.")
    print("\nNext steps:")
    print("1. git remote add origin <your-repo-url>")
    print("2. git push -u origin main")
    print("\nOr start training:")
    print("1. python create_sample_data.py")
    print("2. python train_minimal.py --batch-size 1 --max-steps 500")

if __name__ == "__main__":
    main()
