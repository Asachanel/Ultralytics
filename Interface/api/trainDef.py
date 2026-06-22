from ultralytics import YOLO

from .utils import clear_gpu_memory


def trainDef(task: str, model: str, data: str, imgsz: int, epochs: int) -> str:
    model_obj = None
    try:
        model_path = f'Source/weights/{model}.pt'
        model_obj = YOLO(model_path)

        if data == "mnist160":
            model_obj.train(task=task, data=data, imgsz=imgsz, epochs=epochs, split="test", device=0)
        else:
            model_obj.train(task=task, data=data, imgsz=imgsz, epochs=epochs, device=0)

        return "Training Completed!"

    except RuntimeError as e:
        if "out of memory" in str(e):
            return (
                "Training failed: CUDA out of memory.\n"
                "Suggestions:\n"
                "- Reduce image size (imgsz) to 320 or 416.\n"
                "- Reduce batch size (currently default 16). Consider adding a 'batch' parameter.\n"
                "- Close other GPU-consuming applications."
            )
        else:
            return f"Training failed: {str(e)}"
    except Exception as e:
        return f"Training failed: {str(e)}"
    finally:
        if model_obj is not None:
            del model_obj
        clear_gpu_memory()
