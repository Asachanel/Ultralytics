import gradio as gr

from .api.trainDef import trainDef

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

TASK_IMGSZ_CONFIG = {
    'detect': {'value': 640, 'minimum': 320, 'maximum': 1280, 'step': 32},
    'seg': {'value': 640, 'minimum': 320, 'maximum': 1280, 'step': 32},
    'pose': {'value': 640, 'minimum': 320, 'maximum': 1280, 'step': 32},
    'obb': {'value': 640, 'minimum': 320, 'maximum': 1280, 'step': 32},
    'classify': {'value': 64, 'minimum': 32, 'maximum': 320, 'step': 32},
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


def trainApp():
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
                imgsz = gr.Slider(
                    label="Image Size",
                    minimum=TASK_IMGSZ_CONFIG["detect"]["minimum"],
                    maximum=TASK_IMGSZ_CONFIG["detect"]["maximum"],
                    step=TASK_IMGSZ_CONFIG["detect"]["step"],
                    value=TASK_IMGSZ_CONFIG["detect"]["value"]
                )
                epochs = gr.Slider(
                    label="Epochs",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=30
                )
                train_button = gr.Button(value="Start Training")
                output_box = gr.Textbox(label="Status")

        def update_on_task_change(task):
            models = filter_models_by_task(task)
            default_model = models[0] if models else None
            dataset = TASK_DATA_MAP.get(task, 'coco8.yaml')
            imgsz_cfg = TASK_IMGSZ_CONFIG.get(task, TASK_IMGSZ_CONFIG['detect'])
            return (
                gr.update(choices=models, value=default_model),
                gr.update(choices=[dataset], value=dataset),
                gr.update(value=imgsz_cfg['value'],
                          minimum=imgsz_cfg['minimum'],
                          maximum=imgsz_cfg['maximum'])
            )

        task.change(
            fn=update_on_task_change,
            inputs=task,
            outputs=[model, data, imgsz]
        )

        def train_model(task, model, data, imgsz, epochs):
            return trainDef(task, model, data, imgsz, epochs)

        train_button.click(
            fn=train_model,
            inputs=[task, model, data, imgsz, epochs],
            outputs=output_box
        )
