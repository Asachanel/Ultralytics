import os

import gradio as gr
from dotenv import load_dotenv

from Interface import trainApp, valApp, predictApp, exportApp

load_dotenv("Temp/.env")


def create_app_block() -> gr.Blocks:
    with gr.Blocks() as app_block:
        gr.HTML(
            '''
            <h3 style='text-align: center'>
            <a href='https://github.com/Asachanel/' style='text-decoration: none;' target='_blank'>GitHub</a> |
            <a href='https://blog.asachanel.cn/' style='text-decoration: none;' target='_blank'>MyBlog</a>
            </h3>
            '''
        )
    return app_block


def get_share_preference() -> bool:
    user_input = input("Share application?").strip().lower()
    return user_input in ('y', 'yes', 'true', '1')


def find_favicon_path() -> str | None:
    paths_to_try = [
        "_internal/Temp/Ultralytics.svg",
        "Temp/Ultralytics.svg"
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    return None


def get_auth_credentials():
    auth_env = os.getenv("GRADIO_AUTH", "")
    if not auth_env:
        print("Warning: No authentication set. Consider setting GRADIO_AUTH env var.")
        return None
    credentials = []
    for pair in auth_env.split(","):
        pair = pair.strip()
        if ":" in pair:
            user, pwd = pair.split(":", 1)
            credentials.append((user, pwd))
    return credentials if credentials else None


def main():
    train_block = create_app_block()
    with train_block:
        with gr.Row():
            with gr.Column():
                trainApp()

    val_block = create_app_block()
    with val_block:
        with gr.Row():
            with gr.Column():
                valApp()

    predict_block = create_app_block()
    with predict_block:
        with gr.Row():
            with gr.Column():
                predictApp()

    export_block = create_app_block()
    with export_block:
        with gr.Row():
            with gr.Column():
                exportApp()

    app = gr.TabbedInterface(
        [train_block, val_block, predict_block, export_block],
        ['Train', 'Val', 'Predict', 'Export'],
        title="Ultralytics YOLO",
        theme=gr.themes.Soft()
    )

    share_preference = get_share_preference()
    favicon_path = find_favicon_path()
    auth_creds = get_auth_credentials()

    app.launch(
        pwa=True,
        share=share_preference,
        inline=True,
        inbrowser=True,
        favicon_path=favicon_path,
        auth=auth_creds,
        enable_monitoring=True
    )


if __name__ == '__main__':
    main()
