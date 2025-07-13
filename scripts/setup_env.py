#!/usr/bin/env python3
"""
Environment setup script for SWTBot Fine-tuning Pipeline.
Creates virtual environment and installs dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse


def run_command(command, check=True, shell=False):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        if isinstance(command, str) and not shell:
            command = command.split()
        result = subprocess.run(command, check=check, shell=shell, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")


def check_gpu():
    """Check GPU availability and CUDA support."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU detected: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.1f} GB")
            print(f"CUDA version: {torch.version.cuda}")
            
            if gpu_memory < 3.5:
                print("Warning: GPU memory is less than 4GB. Training may be slow or fail.")
            return True
        else:
            print("Warning: No GPU detected. Training will use CPU (very slow).")
            return False
    except ImportError:
        print("PyTorch not installed yet. GPU check will be performed after installation.")
        return None


def create_virtual_environment(venv_path):
    """Create a Python virtual environment."""
    print(f"Creating virtual environment at {venv_path}")
    
    if venv_path.exists():
        print("Virtual environment already exists.")
        return
    
    run_command([sys.executable, "-m", "venv", str(venv_path)])
    print("Virtual environment created successfully.")


def get_activation_command(venv_path):
    """Get the command to activate the virtual environment."""
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        return f"{activate_script}"
    else:
        activate_script = venv_path / "bin" / "activate"
        return f"source {activate_script}"


def install_dependencies(venv_path, requirements_file):
    """Install dependencies in the virtual environment."""
    print("Installing dependencies...")

    # Get the pip executable in the virtual environment
    if platform.system() == "Windows":
        pip_executable = venv_path / "Scripts" / "pip.exe"
        python_executable = venv_path / "Scripts" / "python.exe"
    else:
        pip_executable = venv_path / "bin" / "pip"
        python_executable = venv_path / "bin" / "python"

    # Upgrade pip first
    run_command([str(pip_executable), "install", "--upgrade", "pip"])

    # Install PyTorch first with CUDA support
    print("Installing PyTorch with CUDA support...")
    pytorch_install_cmd = [
        str(pip_executable), "install", "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]

    try:
        run_command(pytorch_install_cmd)
        print("PyTorch with CUDA 11.8 installed successfully")
    except subprocess.CalledProcessError:
        print("CUDA 11.8 installation failed, trying CUDA 12.1...")
        pytorch_install_cmd = [
            str(pip_executable), "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        try:
            run_command(pytorch_install_cmd)
            print("PyTorch with CUDA 12.1 installed successfully")
        except subprocess.CalledProcessError:
            print("CUDA installation failed, installing CPU-only version...")
            pytorch_install_cmd = [
                str(pip_executable), "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            run_command(pytorch_install_cmd)
            print("PyTorch CPU-only version installed")

    # Install other core ML dependencies
    print("Installing core ML dependencies...")
    core_deps = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0"
    ]

    for dep in core_deps:
        try:
            run_command([str(pip_executable), "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {dep}: {e}")

    # Install remaining requirements
    if requirements_file.exists():
        print("Installing remaining dependencies from requirements.txt...")
        run_command([str(pip_executable), "install", "-r", str(requirements_file)])
    else:
        print(f"Requirements file not found: {requirements_file}")
        # Install essential packages manually
        essential_packages = [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",
            "click>=8.1.0",
            "rich>=13.0.0",
            "psutil>=5.9.0"
        ]
        for package in essential_packages:
            run_command([str(pip_executable), "install", package])

    print("Dependencies installed successfully.")


def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/datasets",
        "models/checkpoints",
        "models/final",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directories created successfully.")


def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """
# Virtual environment
venv/
env/
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data and models
data/raw/
data/processed/
data/datasets/
models/checkpoints/
models/final/
*.bin
*.safetensors

# Logs
logs/
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Cache
.cache/
*.cache
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content.strip())
        print(".gitignore created successfully.")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup SWTBot Fine-tuning Environment")
    parser.add_argument("--venv-name", default="venv", help="Virtual environment name")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU availability check")
    args = parser.parse_args()
    
    print("Setting up SWTBot Fine-tuning Environment...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Setup paths
    project_root = Path.cwd()
    venv_path = project_root / args.venv_name
    requirements_file = project_root / "requirements.txt"
    
    # Create virtual environment
    create_virtual_environment(venv_path)
    
    # Install dependencies
    install_dependencies(venv_path, requirements_file)
    
    # Setup directories
    setup_directories()
    
    # Create .gitignore
    create_gitignore()
    
    # Check GPU (after PyTorch installation)
    if not args.skip_gpu_check:
        # Activate environment and check GPU
        if platform.system() == "Windows":
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            python_executable = venv_path / "bin" / "python"
        
        gpu_check_code = """
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("No GPU detected")
"""
        result = run_command([str(python_executable), "-c", gpu_check_code], check=False)
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate the virtual environment:")
    print(f"   {get_activation_command(venv_path)}")
    print("2. Collect training data:")
    print("   python scripts/collect_data.py")
    print("3. Start training:")
    print("   python scripts/train.py")


if __name__ == "__main__":
    main()
