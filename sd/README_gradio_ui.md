# Stable Diffusion Gradio UI

## 功能
- Base Model 下拉选择：支持 `data/` 下的 `.ckpt` / `.safetensors`。
- LoRA 多选 + 权重表：可同时启用多个 LoRA，并分别设置 `strength`。
- Prompt / Negative Prompt 输入。
- 参数控制：`Sampling Steps` (20-50), `CFG Scale` (7-12), `Seed` (`-1` 随机)。
- 预览输出图像，自动输出 Metadata JSON，并支持下载。

## 文件
- UI 入口：`sd/gradio_ui.py`
- LoRA / 权重加载：`sd/model_loader.py`
- Base 模型转换：`sd/model_converter.py`

## 运行
在项目根目录执行：

```bash
pip install -r requirements.txt
python sd/gradio_ui.py
```

默认地址：`http://127.0.0.1:7860`

## 使用说明
1. 在顶部选择 Base Model。
2. 勾选一个或多个 LoRA，并在权重表中设置每个 LoRA 的 `strength`。
3. 输入 Prompt / Negative Prompt，设置 Steps/CFG/Seed。
4. 点击“生成”。
5. 在右侧查看图片和参数元数据，必要时下载 Metadata JSON。

## 备注
- LoRA 注入当前作用于 U-Net。
- 如 GPU 不可用会回退到 CPU（可在 UI 启动日志中看到 `device`）。
