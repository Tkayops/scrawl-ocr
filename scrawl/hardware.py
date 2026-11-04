"""Use the fastest hardware available: any supported GPU, and every CPU thread.

Imported first by the package, because thread counts must be set before numpy, TensorFlow, PyTorch and
ONNX Runtime start their thread pools.

  SCRAWL_THREADS=8     use this many CPU threads instead of all of them
  SCRAWL_DEVICE=cpu    force a PyTorch device (cpu, cuda, cuda:1, xpu, mps)
"""
import os
import platform
from typing import Tuple

CPU_THREADS = int(os.environ.get('SCRAWL_THREADS') or 0) or os.cpu_count() or 1

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
    os.environ.setdefault(_var, str(CPU_THREADS))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # Apple GPUs: run the odd unsupported op on the CPU


def configure_opencv() -> None:
    import cv2
    cv2.setNumThreads(CPU_THREADS)


def torch_device():
    """Best PyTorch device: NVIDIA (CUDA) or AMD (ROCm) GPU, Intel GPU (XPU), Apple GPU (MPS), else the CPU."""
    import torch
    if os.environ.get('SCRAWL_DEVICE'):
        return torch.device(os.environ['SCRAWL_DEVICE'])
    if torch.cuda.is_available():  # ROCm builds of PyTorch report AMD GPUs here too
        return torch.device('cuda')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def configure_torch_cpu() -> None:
    import torch
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(CPU_THREADS)
    except RuntimeError:  # can only be set once, before any parallel work
        pass


def torch_device_name(device) -> str:
    import torch
    if device.type == 'cuda':
        vendor = 'AMD ROCm' if getattr(torch.version, 'hip', None) else 'NVIDIA CUDA'
        return f'{torch.cuda.get_device_name(device)} ({vendor})'
    if device.type == 'xpu':
        return f'{torch.xpu.get_device_name(device)} (Intel XPU)'
    if device.type == 'mps':
        return f'Apple GPU ({platform.machine()}, Metal)'
    return f'CPU ({CPU_THREADS} threads)'


def onnx_options() -> Tuple[dict, str]:
    """RapidOCR settings: all CPU threads, and a GPU when ONNX Runtime has one (CUDA, or DirectML on Windows)."""
    import onnxruntime as ort
    options = {f'{part}_intra_op_num_threads': CPU_THREADS for part in ('det', 'cls', 'rec')}
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        if hasattr(ort, 'preload_dlls'):
            try:
                ort.preload_dlls()  # lets onnxruntime-gpu use the CUDA libraries that came with PyTorch
            except Exception:
                pass
        options.update({f'{part}_use_cuda': True for part in ('det', 'cls', 'rec')})
        return options, 'NVIDIA CUDA GPU'
    if 'DmlExecutionProvider' in providers:
        options.update({f'{part}_use_dml': True for part in ('det', 'cls', 'rec')})
        return options, 'DirectML GPU'
    return options, f'CPU ({CPU_THREADS} threads)'
