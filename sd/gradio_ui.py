from __future__ import annotations

import json
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch
from PIL import Image
from transformers import CLIPTokenizer

import model_loader
import pipeline


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = ROOT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = CLIPTokenizer(
    vocab_file=str(DATA_DIR / "vocab.json"),
    merges_file=str(DATA_DIR / "merges.txt"),
)

# Runtime cache to avoid reloading base model every click.
RUNTIME = {
    "base_model_name": None,
    "models": None,
    "base_diffusion_state": None,
}

PROMPT_TEMPLATES = {
    "宣传画风格": "Chinese 1950s-1980s propaganda poster, revolutionary poster style, bright flat colors, optimistic composition, detailed illustration",
    "电影感城市": "A cinematic photo of a futuristic city at sunset, volumetric lighting, ultra detailed, sharp focus",
    "国风插画": "Traditional Chinese painting style, ink wash texture, elegant composition, highly detailed",
}


def list_model_files() -> Tuple[List[str], List[str]]:
    if not DATA_DIR.exists():
        return [], []
    base_models = sorted([p.name for p in DATA_DIR.glob("*.ckpt")]) + sorted([p.name for p in DATA_DIR.glob("*.safetensors")])
    lora_models = sorted([p.name for p in DATA_DIR.glob("*.safetensors")]) + sorted([p.name for p in DATA_DIR.glob("*.ckpt")])
    return base_models, lora_models


def _restore_base_diffusion_weights() -> None:
    if RUNTIME["models"] is None or RUNTIME["base_diffusion_state"] is None:
        return

    diffusion = RUNTIME["models"]["diffusion"]
    restored_state = {
        k: v.to(device=DEVICE, dtype=diffusion.state_dict()[k].dtype)
        for k, v in RUNTIME["base_diffusion_state"].items()
    }
    diffusion.load_state_dict(restored_state, strict=True)


def load_base_model(base_model_name: str) -> str:
    if not base_model_name:
        raise gr.Error("请选择 Base Model")

    ckpt_path = DATA_DIR / base_model_name
    if not ckpt_path.exists():
        raise gr.Error(f"Base model not found: {ckpt_path}")

    models = model_loader.preload_models_from_standard_weights(str(ckpt_path), DEVICE)
    RUNTIME["base_model_name"] = base_model_name
    RUNTIME["models"] = models
    RUNTIME["base_diffusion_state"] = {
        k: v.detach().cpu().clone() for k, v in models["diffusion"].state_dict().items()
    }

    return f"Base model loaded on {DEVICE}: {base_model_name}"


def parse_lora_weights_table(table_data, selected_loras: List[str]) -> Dict[str, float]:
    selected_set = set(selected_loras or [])
    weight_map: Dict[str, float] = {}

    if table_data:
        for row in table_data:
            if not row or len(row) < 2:
                continue
            name = str(row[0]).strip()
            if not name:
                continue
            try:
                weight = float(row[1])
            except Exception:
                continue
            weight_map[name] = weight

    # Keep only selected LoRAs, default weight=1.0
    return {name: weight_map.get(name, 1.0) for name in selected_set}


def apply_loras_to_current_unet(lora_weight_map: Dict[str, float]) -> List[dict]:
    _restore_base_diffusion_weights()

    applied = []
    for lora_name, lora_weight in lora_weight_map.items():
        lora_path = DATA_DIR / lora_name
        if not lora_path.exists():
            continue

        info = model_loader.load_lora_weights_into_unet(
            RUNTIME["models"]["diffusion"].unet,
            checkpoint_path=str(lora_path),
            device=DEVICE,
            default_alpha=1.0,
            lora_scale=lora_weight,
        )
        applied.append(
            {
                "name": lora_name,
                "weight": lora_weight,
                "num_loaded": info["num_loaded"],
                "num_skipped": info["num_skipped"],
            }
        )
    return applied


def generate_image(
    base_model_name: str,
    selected_loras: List[str],
    lora_weights_table,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg_scale: float,
    seed: int,
):
    if not prompt.strip():
        raise gr.Error("Prompt 不能为空")

    if RUNTIME["models"] is None or RUNTIME["base_model_name"] != base_model_name:
        load_base_model(base_model_name)

    use_seed = random.randint(0, 2**31 - 1) if seed < 0 else int(seed)
    lora_weight_map = parse_lora_weights_table(lora_weights_table, selected_loras)
    applied_loras = apply_loras_to_current_unet(lora_weight_map)

    image_array = pipeline.generate(
        prompt=prompt,
        uncond_prompt=negative_prompt,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=float(cfg_scale),
        sampler_name="ddpm",
        n_inference_steps=int(steps),
        models=RUNTIME["models"],
        seed=use_seed,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=TOKENIZER,
    )

    image = Image.fromarray(image_array)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = IMAGES_DIR / f"gradio_{timestamp}.png"
    image.save(image_path)

    metadata = {
        "timestamp": timestamp,
        "device": DEVICE,
        "base_model": base_model_name,
        "loras": applied_loras,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampling_steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "seed": int(use_seed),
        "output_image": str(image_path),
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        metadata_path = f.name

    return image, metadata, metadata_path


def fill_prompt(template_name: str) -> str:
    return PROMPT_TEMPLATES.get(template_name, "")


def refresh_choices():
    base_models, lora_models = list_model_files()
    table_rows = [[name, 1.0] for name in lora_models]
    return (
        gr.update(choices=base_models, value=base_models[0] if base_models else None),
        gr.update(choices=lora_models, value=[]),
        gr.update(value=table_rows),
    )


def build_app() -> gr.Blocks:
    base_models, lora_models = list_model_files()

    with gr.Blocks(title="Stable Diffusion Gradio UI") as demo:
        gr.Markdown("## Stable Diffusion 推理界面 (Gradio)")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 模型选择区")
                base_model = gr.Dropdown(
                    label="Base Model (.ckpt / .safetensors)",
                    choices=base_models,
                    value=base_models[0] if base_models else None,
                )
                lora_selector = gr.CheckboxGroup(
                    label="LoRA Selector (可多选)",
                    choices=lora_models,
                    value=[],
                )
                lora_weights_table = gr.Dataframe(
                    headers=["lora_name", "strength"],
                    datatype=["str", "number"],
                    value=[[name, 1.0] for name in lora_models],
                    row_count=(max(1, len(lora_models)), "fixed"),
                    col_count=(2, "fixed"),
                    label="LoRA 权重表（只对勾选项生效）",
                )

                with gr.Row():
                    refresh_btn = gr.Button("刷新模型列表")
                    load_base_btn = gr.Button("加载 Base Model")

                load_status = gr.Textbox(label="模型加载状态", interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 输入区")
                prompt = gr.Textbox(label="Prompt", lines=6)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    value="blurry, low quality, distorted, bad anatomy, artifacts",
                )

                template_name = gr.Dropdown(
                    label="常用提示词模板",
                    choices=list(PROMPT_TEMPLATES.keys()),
                    value="宣传画风格",
                )
                apply_template_btn = gr.Button("应用模板到 Prompt")

                steps = gr.Slider(label="Sampling Steps", minimum=20, maximum=50, value=30, step=1)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=7.0, maximum=12.0, value=8.0, step=0.1)
                seed = gr.Number(label="Seed (-1 为随机)", value=-1, precision=0)
                generate_btn = gr.Button("生成", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 预览区")
                output_image = gr.Image(label="生成结果", type="pil")
                metadata_json = gr.JSON(label="Metadata（点击图片后可在此查看参数）")
                metadata_file = gr.File(label="导出 Metadata JSON")

        refresh_btn.click(refresh_choices, outputs=[base_model, lora_selector, lora_weights_table])
        load_base_btn.click(load_base_model, inputs=[base_model], outputs=[load_status])
        apply_template_btn.click(fill_prompt, inputs=[template_name], outputs=[prompt])

        generate_btn.click(
            generate_image,
            inputs=[base_model, lora_selector, lora_weights_table, prompt, negative_prompt, steps, cfg_scale, seed],
            outputs=[output_image, metadata_json, metadata_file],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
