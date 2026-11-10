"""Import CPU-side scientific modules once in a CUDA-free fork server."""
import os
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK','1')
import torch
from scripts.cross_dataset_campaign import msd_mixed_losses

if torch.cuda.is_initialized():
    raise RuntimeError('The import server must never initialize CUDA')
