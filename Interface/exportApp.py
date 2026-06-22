import gradio as gr

from .api.exportDef import exportDef

TASK_CHOICES = ['detect', 'seg', 'classify', 'pose', 'obb']

MODEL_CHOICES = [
    'yolo26n', 'yolo26n-cls', 'yolo26n-obb', 'yolo26n-pose', 'yolo26n-seg',
    'yolo26s', 'yolo26s-cls', 'yolo26s-obb', 'yolo26s-pose', 'yolo26s-seg',
    'yolo26m', 'yolo26m-cls', 'yolo26m-obb', 'yolo26m-pose', 'yolo26m-seg',
    'yolo26l', 'yolo26l-cls', 'yolo26l-obb', 'yolo26l-pose', 'yolo26l-seg',
    'yolo26x', 'yolo26x-cls', 'yolo26x-obb', 'yolo26x-pose', 'yolo26x-seg',
]


def filter_models_by_task(task: str):
    if task == 'detect':
        return [m for m in MODEL_CHOICES if not any(suffix in m for suffix in ['-cls', '-obb', '-pose', '-seg'])]
    elif task == 'seg':
        return [m for m in MODEL_CHOICES if m.endswith('-seg')]
    elif task == 'classify':
        return [m for m in MODEL_CHOICES if m.endswith('-cls')]
    elif task == 'pose':
        return [m for m in MODEL_CHOICES if m.endswith('-pose')]
    elif task == 'obb':
        return [m for m in MODEL_CHOICES if m.endswith('-obb')]
    else:
        return MODEL_CHOICES


def exportApp():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                task = gr.Dropdown(
                    label="Choose a task",
                    choices=TASK_CHOICES,
                    value="detect"
                )
                model = gr.Dropdown(
                    label="Model",
                    choices=filter_models_by_task("detect"),
                    value="yolo26n"
                )
                export_button = gr.Button("Export to ONNX")
                output_box = gr.Textbox(label="Status")

        def update_models_by_task(task):
            models = filter_models_by_task(task)
            default_model = models[0] if models else None
            return gr.update(choices=models, value=default_model)

        task.change(
            fn=update_models_by_task,
            inputs=task,
            outputs=model
        )

        def export_model(model):
            return exportDef(model)

        export_button.click(
            fn=export_model,
            inputs=model,
            outputs=output_box
        )
