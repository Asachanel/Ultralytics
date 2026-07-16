# Ultralytics YOLO - Gradio 接口

这是一个基于 Ultralytics YOLO 的个人开发演示项目，使用 Gradio 提供可交互的 Web 界面，用于模型训练（Train）、验证（Val）、推理（Predict）和导出（Export）。适用于希望在本地或带 GPU 的服务器上快速试用 Ultralytics 模型的开发者。

## 特性
- 使用 ultralytics 库加载与训练 YOLO 模型
- 提供 Gradio 三方界面：训练、验证、推理与导出为 ONNX
- 支持图片与视频推理，视频推理会尝试保留原始音频（依赖 FFmpeg）
- 基本的 GPU 内存清理与 LRU 模型缓存优化

## 技术栈
- 语言：Python 3
- 框架 / 运行时：Gradio（UI）、Ultralytics（模型）、PyTorch（后端）
- 重要依赖（示例）：ultralytics, torch, gradio, opencv-python, ffmpeg-python, imageio-ffmpeg

## 项目结构（顶层）
```
app.py                   # 程序入口，创建并启动 Gradio 多选项卡界面
environment.yml          # Conda 环境配置（建议通过 conda 创建环境）
requirements.txt         # pip 依赖（注意：当前文件存在编码/格式问题，推荐使用 environment.yml）
Interface/               # Gradio 界面与 API 封装
  api/                   # 后端功能实现：trainDef, valDef, predictDef, exportDef, utils
  trainApp.py            # "Train" 选项卡的界面定义
  valApp.py              # "Val" 选项卡的界面定义
  predictApp.py          # "Predict" 选项卡的界面定义
  exportApp.py           # "Export" 选项卡的界面定义
Source/                  # 输入输出目录占位（数据、图片、视频与保存的权重）
  weights/               # 模型权重（.pt）目录（需手动放入或训练生成）
  inputs/                # 推理输入目录（运行时创建）
  outputs/               # 推理输出目录（运行时创建）
Temp/                    # 临时/环境文件（包含示例 .env）
```

## 快速开始（推荐）
1. 克隆仓库：
```bash
git clone https://github.com/Asachanel/Ultralytics.git
cd Ultralytics
```

2. 建议使用 Conda（environment.yml 已配置）：
```bash
conda env create -f environment.yml
conda activate Ultralytics
```
如果你不使用 conda，可以手动安装 pip 依赖，但注意 requirements.txt 可能有编码问题，优先参考 environment.yml 中的 pip 部分。

3. 配置认证（可选但推荐用于保护 Gradio 应用）：
- 仓库内示例：Temp/.env，包含 `GRADIO_AUTH=USER:PASSWORD`。你可以修改该文件或在运行前导出环境变量：
```bash
export GRADIO_AUTH="user:password"
# Windows (PowerShell)
# setx GRADIO_AUTH "user:password"
```

4. 安装并确保系统上存在 FFmpeg（用于视频推理时合并音频）：
- 在 Linux/macOS：`sudo apt install ffmpeg` 或 `brew install ffmpeg`
- Windows：请将 ffmpeg 可执行文件加入 PATH。

5. 运行应用：
```bash
python app.py
```
运行时会询问 `Share application?`（是否启用 Gradio 的 share 链接），输入 `y`/`n`。应用将在浏览器中打开并显示四个选项卡：Train、Val、Predict、Export。

## 使用说明要点
- 模型权重：请把你要使用/训练的 `.pt` 文件放到 `Source/weights/` 目录下，界面中通过下拉选择模型名称（例如 `yolo26n` 会对应 `Source/weights/yolo26n.pt`）。
- 推理（图片）：上传图片并选择参数，将返回标注后的图片。
- 推理（视频）：上传视频（或提供路径），程序会处理每帧并生成一个临时无声 MP4，然后尝试用 FFmpeg 合并原始音频。若 FFmpeg 不可用或合并失败，程序会回退为仅视频输出。
- 导出：Export 选项卡调用 ultralytics 的 export 功能导出 ONNX。

## 常见问题与排查
- CUDA OOM（显存不足）：Train 中捕获了常见的 OOM 错误并给出建议（减小 imgsz、减小 batch、关闭其他占用 GPU 的程序）。
- requirements.txt 编码异常：仓库里的 requirements.txt 文件似乎包含奇怪的字符，建议使用 environment.yml 或直接从 environment.yml 的 pip 列表安装。
- FFmpeg 报错：检查系统中 ffmpeg 是否安装且可执行，或查看 predictDef.py 中 `_merge_audio_video` 的 stderr 输出以排查原因。

## 开发与贡献
- 代码风格：简单明了，主要分为 UI 层（Interface/*.py）与功能实现层（Interface/api/*.py）。
- 如果要增加新模型或任务：更新 `MODEL_CHOICES` 与 `TASK_CHOICES`，并在 `Source/weights/` 加入对应权重文件。
- 欢迎提交 issue 或 pull request。若需联系作者，可访问 app.py 中的链接：GitHub (https://github.com/Asachanel/) 或博客 (https://blog.asachanel.cn/)。

## 致谢
基于 Ultralytics YOLO 与 Gradio 搭建示例界面，适合作为教学或快速原型工具。

