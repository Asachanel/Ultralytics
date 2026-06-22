import gc
import warnings

import torch


def clear_gpu_memory():
    """彻底清理 GPU 显存，增强异常处理。"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()  # 等待所有 CUDA 内核完成
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA error" in str(e):
                warnings.warn(f"GPU memory cleanup skipped due to CUDA error: {e}")
                # 尝试重置峰值统计，不抛出异常
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass
            else:
                raise
