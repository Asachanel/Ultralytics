from warnings import simplefilter, catch_warnings

from ultralytics import YOLO

from .utils import clear_gpu_memory


def exportDef(model: str) -> str:
    model_obj = None
    try:
        model_path = f'Source/weights/{model}.pt'
        model_obj = YOLO(model_path)
        # noinspection PyTypeChecker
        with catch_warnings():
            simplefilter('ignore')
            model_obj.export(format="onnx", device=0)
        return "Export Completed!"
    except Exception as e:
        return f"Export failed: {str(e)}"
    finally:
        if model_obj is not None:
            del model_obj
        clear_gpu_memory()
