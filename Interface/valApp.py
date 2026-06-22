import gradio as gr

from .api.valDef import valDef

TASK_CHOICES = ['detect', 'seg', 'classify', 'pose', 'obb']

MODEL_CHOICES = [
    'yolo26n', 'yolo26n-cls', 'yolo26n-obb', 'yolo26n-pose', 'yolo26n-seg',
    'yolo26s', 'yolo26s-cls', 'yolo26s-obb', 'yolo26s-pose', 'yolo26s-seg',
    'yolo26m', 'yolo26m-cls', 'yolo26m-obb', 'yolo26m-pose', 'yolo26m-seg',
    'yolo26l', 'yolo26l-cls', 'yolo26l-obb', 'yolo26l-pose', 'yolo26l-seg',
    'yolo26x', 'yolo26x-cls', 'yolo26x-obb', 'yolo26x-pose', 'yolo26x-seg',
]

TASK_DATA_MAP = {
    'detect': 'coco8.yaml',
    'seg': 'coco8-seg.yaml',
    'classify': 'mnist160',
    'pose': 'coco8-pose.yaml',
    'obb': 'dota8.yaml',
}


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


def valApp():
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
                data = gr.Dropdown(
                    label="Data YAML Path",
                    choices=[TASK_DATA_MAP["detect"]],
                    value=TASK_DATA_MAP["detect"]
                )
                val_button = gr.Button("Start Validation")
                output_box = gr.Textbox(label="Status")

        def update_on_task_change(task):
            models = filter_models_by_task(task)
            default_model = models[0] if models else None
            dataset = TASK_DATA_MAP.get(task, 'coco8.yaml')
            return (
                gr.update(choices=models, value=default_model),
                gr.update(choices=[dataset], value=dataset)
            )

        task.change(
            fn=update_on_task_change,
            inputs=task,
            outputs=[model, data]
        )

        def val_model(model, data):
            return valDef(model, data)

        val_button.click(
            fn=val_model,
            inputs=[model, data],
            outputs=output_box
        )
