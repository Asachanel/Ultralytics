import gradio as gr

from .api.predictDef import predictDef

TASK_CHOICES = ['detect', 'seg', 'classify', 'pose', 'obb']

MODEL_CHOICES = [
    'yolo26n', 'yolo26n-cls', 'yolo26n-obb', 'yolo26n-pose', 'yolo26n-seg',
    'yolo26s', 'yolo26s-cls', 'yolo26s-obb', 'yolo26s-pose', 'yolo26s-seg',
    'yolo26m', 'yolo26m-cls', 'yolo26m-obb', 'yolo26m-pose', 'yolo26m-seg',
    'yolo26l', 'yolo26l-cls', 'yolo26l-obb', 'yolo26l-pose', 'yolo26l-seg',
    'yolo26x', 'yolo26x-cls', 'yolo26x-obb', 'yolo26x-pose', 'yolo26x-seg',
]

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


def predictApp():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type"
                )
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
                imgsz = gr.Slider(
                    label="Image Size",
                    minimum=TASK_IMGSZ_CONFIG["detect"]["minimum"],
                    maximum=TASK_IMGSZ_CONFIG["detect"]["maximum"],
                    step=TASK_IMGSZ_CONFIG["detect"]["step"],
                    value=TASK_IMGSZ_CONFIG["detect"]["value"]
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25
                )
                predict_button = gr.Button(value="Detect Objects")
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video]
        )

        def update_on_task_change(task):
            models = filter_models_by_task(task)
            default_model = models[0] if models else None
            imgsz_cfg = TASK_IMGSZ_CONFIG.get(task, TASK_IMGSZ_CONFIG['detect'])
            return (
                gr.update(choices=models, value=default_model),
                gr.update(value=imgsz_cfg['value'],
                          minimum=imgsz_cfg['minimum'],
                          maximum=imgsz_cfg['maximum'])
            )

        task.change(
            fn=update_on_task_change,
            inputs=task,
            outputs=[model, imgsz]
        )

        def predict_model(image, video, model, imgsz, conf_threshold, input_type):
            if input_type == "Image":
                return predictDef(image=image, model=model, imgsz=imgsz, conf_threshold=conf_threshold)
            else:
                return predictDef(video=video, model=model, imgsz=imgsz, conf_threshold=conf_threshold)

        predict_button.click(
            fn=predict_model,
            inputs=[image, video, model, imgsz, conf_threshold, input_type],
            outputs=[output_image, output_video]
        )
