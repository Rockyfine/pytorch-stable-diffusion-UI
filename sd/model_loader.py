from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from attention import SelfAttention
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion, inject_lora_into_linear

import model_converter

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:  # pragma: no cover - handled at runtime with a clear error message
    load_safetensors_file = None


_LORA_UP_SUFFIXES = (".lora_up.weight", ".lora_A.weight")
_LORA_DOWN_SUFFIXES = (".lora_down.weight", ".lora_B.weight")
_LORA_ALPHA_SUFFIXES = (".alpha", ".lora_alpha")
_COMMON_LORA_PREFIXES = (
    "unet.",
    "model.diffusion_model.",
    "diffusion_model.",
)


def _load_state_dict_from_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError(
                "Loading .safetensors LoRA weights requires the 'safetensors' package. "
                "Install it with: pip install safetensors"
            )
        checkpoint = load_safetensors_file(str(checkpoint_path), device=str(device))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if "lora_state_dict" in checkpoint and isinstance(checkpoint["lora_state_dict"], dict):
            return checkpoint["lora_state_dict"]
    return checkpoint


def _normalize_lora_key(key):
    for prefix in _COMMON_LORA_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _get_alpha_value(state_dict, normalized_prefix, raw_prefixes, default_alpha):
    candidate_prefixes = [normalized_prefix]
    candidate_prefixes.extend(sorted(raw_prefixes, key=len, reverse=True))

    for prefix in candidate_prefixes:
        for suffix in _LORA_ALPHA_SUFFIXES:
            alpha_key = prefix + suffix
            if alpha_key in state_dict:
                alpha_value = state_dict[alpha_key]
                if isinstance(alpha_value, torch.Tensor):
                    return float(alpha_value.item())
                return float(alpha_value)
    return float(default_alpha)


def _split_lora_state_dict(state_dict):
    pairs = defaultdict(dict)
    alphas = {}
    raw_prefixes = defaultdict(set)

    for key, value in state_dict.items():
        normalized_key = _normalize_lora_key(key)

        for suffix in _LORA_UP_SUFFIXES:
            if normalized_key.endswith(suffix):
                prefix = normalized_key[:-len(suffix)]
                pairs[prefix]["up"] = value
                raw_prefixes[prefix].add(key[:-len(suffix)])
                break
        else:
            for suffix in _LORA_DOWN_SUFFIXES:
                if normalized_key.endswith(suffix):
                    prefix = normalized_key[:-len(suffix)]
                    pairs[prefix]["down"] = value
                    raw_prefixes[prefix].add(key[:-len(suffix)])
                    break

    for prefix in list(pairs.keys()):
        alphas[prefix] = _get_alpha_value(state_dict, prefix, raw_prefixes[prefix], 1.0)

    return pairs, alphas


def _apply_lora_to_self_attention(module, qkv_deltas):
    if not isinstance(module, SelfAttention):
        raise TypeError(f"Expected SelfAttention, got {type(module)!r}")

    in_proj = module.in_proj
    expected_out_features = in_proj.out_features
    expected_in_features = in_proj.in_features
    if expected_out_features % 3 != 0:
        raise ValueError("SelfAttention.in_proj must have 3 * d_embed output features")

    d_embed = expected_out_features // 3
    merged_delta = torch.zeros_like(in_proj.weight)

    for part_name, part_delta in qkv_deltas.items():
        if part_name == "q_proj":
            start_index = 0
        elif part_name == "k_proj":
            start_index = 1
        elif part_name == "v_proj":
            start_index = 2
        else:
            raise ValueError(f"Unsupported SelfAttention LoRA part: {part_name}")

        if part_delta.shape != (d_embed, expected_in_features):
            raise ValueError(
                f"SelfAttention part delta for {part_name} must have shape {(d_embed, expected_in_features)}, "
                f"got {tuple(part_delta.shape)}"
            )

        row_start = start_index * d_embed
        row_end = row_start + d_embed
        merged_delta[row_start:row_end] = part_delta.to(device=in_proj.weight.device, dtype=in_proj.weight.dtype)

    with torch.no_grad():
        in_proj.weight.add_(merged_delta)


def load_lora_weights_into_unet(unet, checkpoint_path, device="cpu", default_alpha=1.0, lora_scale=1.0):
    """Load LoRA weights from a checkpoint and inject them into a U-Net.

    Supported key patterns:
    - <module_path>.lora_up.weight / <module_path>.lora_down.weight
    - <module_path>.lora_A.weight / <module_path>.lora_B.weight
    - optional <module_path>.alpha / <module_path>.lora_alpha

    For SelfAttention, q_proj/k_proj/v_proj entries are merged into the combined
    in_proj weight. CrossAttention and regular Linear layers are updated directly.
    """

    state_dict = _load_state_dict_from_checkpoint(checkpoint_path, device)
    lora_pairs, alphas = _split_lora_state_dict(state_dict)

    modules = dict(unet.named_modules())
    loaded_modules = []
    skipped_prefixes = []
    grouped_self_attention = defaultdict(dict)

    for prefix, parts in lora_pairs.items():
        if "up" not in parts or "down" not in parts:
            skipped_prefixes.append(prefix)
            continue

        module = modules.get(prefix)
        alpha = alphas.get(prefix, default_alpha) * float(lora_scale)

        if isinstance(module, nn.Linear):
            up_weight = parts["up"]
            down_weight = parts["down"]
            rank = up_weight.shape[1]
            inject_lora_into_linear(module, up_weight, down_weight, r=rank, alpha=alpha)
            loaded_modules.append(prefix)
            continue

        if prefix.endswith(("q_proj", "k_proj", "v_proj")):
            # prefix may be like "... .q_proj" or just "q_proj" depending on checkpoint key normalization.
            part_name = None
            parent_name = None
            if "." in prefix:
                parent_name, part_name = prefix.rsplit(".", 1)
            else:
                part_name = prefix
                # try to resolve the parent module by matching module names that end with the part name
                candidates = [n for n in modules.keys() if n.split(".")[-1] == part_name]
                if len(candidates) == 1:
                    parent_name = candidates[0]
                else:
                    # try a looser match: module names that end with ".{part_name}"
                    candidates = [n for n in modules.keys() if n.endswith("." + part_name)]
                    if len(candidates) == 1:
                        parent_name = candidates[0]
                    else:
                        # ambiguous or not found -- skip this prefix
                        skipped_prefixes.append(prefix)
                        continue

            parent_module = modules.get(parent_name)
            if isinstance(parent_module, SelfAttention):
                up_weight = parts["up"]
                down_weight = parts["down"]
                rank = up_weight.shape[1]
                grouped_self_attention[parent_name][part_name] = ((alpha / rank) * (up_weight @ down_weight))
                continue

        skipped_prefixes.append(prefix)

    for parent_name, part_deltas in grouped_self_attention.items():
        parent_module = modules[parent_name]
        _apply_lora_to_self_attention(parent_module, part_deltas)
        loaded_modules.append(parent_name)

    return {
        "loaded_modules": loaded_modules,
        "skipped_prefixes": skipped_prefixes,
        "num_loaded": len(loaded_modules),
        "num_skipped": len(skipped_prefixes),
    }

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }