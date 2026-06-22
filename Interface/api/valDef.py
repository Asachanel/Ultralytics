from ultralytics import YOLO

from .utils import clear_gpu_memory


def valDef(model: str, data: str) -> str:
    model_obj = None
    try:
        model_path = f'Source/weights/{model}.pt'
        model_obj = YOLO(model_path)

        if data == "mnist160":
            model_obj.val(data=data, split="test", device=0)
        else:
            model_obj.val(data=data, device=0)

        return "Validation Completed!"
    except Exception as e:
        return f"Validation failed: {str(e)}"
    finally:
        if model_obj is not None:
            del model_obj
        clear_gpu_memory()
