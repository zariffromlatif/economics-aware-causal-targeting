"""Hardware acceleration and device management utilities.

Auto-detects high-performance hardware:
- CPU cores (e.g. Intel i9-14900K 24C/32T)
- NVIDIA CUDA GPUs (e.g. RTX 4090 24GB VRAM)
- Memory configuration (e.g. 64GB DDR5)
Seamlessly switches estimators between CPU and CUDA without requiring manual code changes.
"""

import os
from typing import Dict, Any, Tuple


def get_hardware_info() -> Dict[str, Any]:
    """Inspect local hardware and CUDA availability."""
    cpu_count = os.cpu_count() or 4
    gpu_available = False
    gpu_name = "None"
    gpu_mem_gb = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        # Check through nvidia-smi if torch not yet installed in active venv
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().split("\n")
                first = lines[0].split(",")
                gpu_available = True
                gpu_name = first[0].strip()
                if len(first) > 1:
                    gpu_mem_gb = float(first[1].strip().replace("MiB", "")) / 1024.0
        except Exception:
            pass

    return {
        "cpu_count": cpu_count,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_mem_gb": round(gpu_mem_gb, 2),
        "recommended_device": "cuda" if gpu_available else "cpu",
        "recommended_n_jobs": cpu_count,
    }


def get_xgboost_device_params(device: str = "auto") -> Dict[str, Any]:
    """Return optimal XGBoost device and tree method parameters."""
    if device == "auto":
        hw = get_hardware_info()
        target_device = hw["recommended_device"]
    else:
        target_device = device.lower()

    if target_device == "cuda":
        return {
            "device": "cuda",
            "tree_method": "hist",
        }
    else:
        return {
            "device": "cpu",
            "tree_method": "hist",
            "n_jobs": os.cpu_count() or 4,
        }


if __name__ == "__main__":
    info = get_hardware_info()
    print("Detected Hardware Configuration:")
    for k, v in info.items():
        print(f"  {k}: {v}")
