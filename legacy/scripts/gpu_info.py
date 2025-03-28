import torch
import subprocess
import os
import sys

def print_gpu_info():
    """Print detailed information about GPU availability and usage."""
    print("\n" + "="*50)
    print("GPU INFORMATION")
    print("="*50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"Current Device Index: {current_device}")
        print(f"Current Device Name: {torch.cuda.get_device_name(current_device)}")
        
        # Check if tensor operations are actually using CUDA
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        try:
            cuda_tensor = test_tensor.cuda()
            print(f"Tensor successfully moved to CUDA: {cuda_tensor.device}")
        except Exception as e:
            print(f"Error moving tensor to CUDA: {e}")
            
        # Memory information
        try:
            print(f"\nMemory allocated: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated(current_device) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error getting memory info: {e}")
    else:
        print("CUDA is not available. Training will use CPU.")
    
    # Try to get nvidia-smi output
    print("\n" + "-"*50)
    print("NVIDIA-SMI OUTPUT")
    print("-"*50)
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        print(nvidia_smi)
    except:
        print("nvidia-smi command not available or failed.")
        
    print("="*50)

if __name__ == "__main__":
    print_gpu_info() 