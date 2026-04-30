# Environment check script for the CWRU bearing fault diagnosis project.

import os
import platform
import sys


def main() -> None:
    print(f"Python path: {sys.executable}")
    print(f"Python version: {platform.python_version()}")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch version: Not installed")
        print("CUDA available: False")

    print(f"Current working directory: {os.getcwd()}")


if __name__ == "__main__":
    main()

