import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices (GPUs)
    num_cuda_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices (GPUs) available: {num_cuda_devices}")
    
    # Get the name of the current CUDA device
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current CUDA device name: {current_device_name}")
    
    # Check if PyTorch is currently running on a CUDA device
    if torch.cuda.current_device() == 0:
        print("PyTorch is running on GPU.")
    else:
        print("PyTorch is running on a different GPU.")
else:
    print("CUDA (GPU support) is not available. PyTorch will run on CPU.")
