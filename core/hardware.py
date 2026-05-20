from __future__ import annotations

import psutil

try:
    import torch
except ImportError:
    torch = None


def detect_hardware() -> dict:
    """Detect the best inference device.

    Backend resolution order:
      1. CUDA (NVIDIA) — torch.cuda.is_available() with torch.version.hip is None
      2. ROCm (AMD)    — torch.cuda.is_available() with torch.version.hip set
                          (PyTorch-ROCm reuses the CUDA namespace, so the device
                          string is still "cuda:0" — Ultralytics handles it natively)
      3. MPS  (Apple)  — torch.backends.mps.is_available()
      4. CPU           — fallback
    """
    info = {
        "device": "cpu",
        "device_name": "CPU",
        "backend": "cpu",
        "vram_gb": 0.0,
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "cpu_count": psutil.cpu_count(logical=False) or 1,
        "recommended_batch": 4,
        "torch_available": torch is not None,
        "gpu_available": False,
    }

    if torch is None:
        info["recommended_batch"] = max(2, min(8, info["cpu_count"]))
        return info

    if torch.cuda.is_available():
        is_rocm = bool(getattr(torch.version, "hip", None))
        info["device"] = "cuda:0"
        info["backend"] = "rocm" if is_rocm else "cuda"
        info["gpu_available"] = True
        info["device_name"] = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info["vram_gb"] = round(vram_gb, 1)
        batch = _batch_for_vram(vram_gb)
        # ROCm in WSL2 uses extra system RAM for the HSA runtime bridge;
        # halve the batch to leave headroom and avoid OOM crashes.
        if is_rocm:
            batch = max(2, batch // 2)
        info["recommended_batch"] = batch
        return info

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["backend"] = "mps"
        info["gpu_available"] = True
        info["device_name"] = "Apple Silicon GPU"
        # MPS shares system RAM; size batch off RAM with a conservative cap.
        info["vram_gb"] = info["ram_gb"]
        info["recommended_batch"] = min(16, _batch_for_vram(info["ram_gb"] / 2))
        return info

    info["recommended_batch"] = max(2, min(8, info["cpu_count"]))
    return info


def _batch_for_vram(vram_gb: float) -> int:
    if vram_gb >= 20:
        return 32
    if vram_gb >= 12:
        return 24
    if vram_gb >= 8:
        return 16
    if vram_gb >= 6:
        return 8
    if vram_gb >= 4:
        return 4
    return 2


_BACKEND_LABEL = {
    "cuda": "NVIDIA (CUDA)",
    "rocm": "AMD (ROCm)",
    "mps":  "Apple (MPS)",
    "cpu":  "CPU",
}


def format_hardware(info: dict) -> str:
    backend = info.get("backend", "cpu")
    label = _BACKEND_LABEL.get(backend, backend)
    if info["gpu_available"]:
        if backend == "mps":
            mem_line = f"Shared RAM: {info['ram_gb']} GB"
        else:
            mem_line = f"VRAM: {info['vram_gb']} GB"
        return (
            f"{label}: {info['device_name']}\n"
            f"{mem_line}\n"
            f"System RAM: {info['ram_gb']} GB\n"
            f"Recommended batch: {info['recommended_batch']}"
        )
    return (
        f"Device: CPU ({info['cpu_count']} cores)\n"
        f"RAM: {info['ram_gb']} GB\n"
        f"Recommended batch: {info['recommended_batch']}\n"
        f"(no GPU detected — inference will be slow)"
    )
