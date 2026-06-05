import json
import os
import re
from typing import Dict, List, Optional, Set

import comfy.lora_convert
import comfy.sd
import comfy.utils
import folder_paths
import torch
from safetensors.torch import save_file


def _clone_lora_state(lora: Dict[str, object]) -> Dict[str, object]:
    cloned: Dict[str, object] = {}
    for key, value in lora.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def _parse_layer_indices(indices_text: str) -> List[int]:
    text = (indices_text or "").strip()
    if not text:
        return []

    parsed: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                raise ValueError(f"invalid range '{token}': start > end")
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(int(token))
    return parsed


def _extract_layer_id(match: re.Match) -> Optional[int]:
    if match.groups():
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            pass

    full = match.group(0)
    number = re.search(r"\d+", full)
    if number:
        return int(number.group(0))
    return None


_LORA_PAIR_SUFFIXES = (
    (".lora_up.weight", ".lora_down.weight", ".lora_mid.weight"),
    ("_lora.up.weight", "_lora.down.weight", None),
    (".lora_B.weight", ".lora_A.weight", None),
    (".lora.up.weight", ".lora.down.weight", None),
    (".lora_B", ".lora_A", None),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight", None),
    (".lora_B.default.weight", ".lora_A.default.weight", None),
)


def _make_rank_tensor(value: object, rank: int, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.tensor(float(rank), dtype=value.dtype, device=value.device)
    return torch.tensor(float(rank), dtype=ref.dtype, device=ref.device)


def _make_scalar_tensor(value: object, scalar: float, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.tensor(float(scalar), dtype=value.dtype, device=value.device)
    return torch.tensor(float(scalar), dtype=ref.dtype, device=ref.device)


def _extract_lora_alpha(lora_sd: Dict[str, object], alpha_key: str, down_tensor: torch.Tensor) -> float:
    alpha = lora_sd.get(alpha_key)
    if alpha is None:
        return float(down_tensor.shape[0])
    if isinstance(alpha, torch.Tensor):
        return float(alpha.item())
    return float(alpha)


def _resolve_compute_device(device_name: str) -> torch.device:
    name = (device_name or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("compute_device='cuda' but CUDA is not available")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported compute_device '{device_name}'")


def _scan_mergeable_lora_pairs(lora_sd: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    pairs: Dict[str, Dict[str, object]] = {}
    recognized_tensor_keys: Set[str] = set()

    for up_suffix, down_suffix, mid_suffix in _LORA_PAIR_SUFFIXES:
        for key in lora_sd:
            if not key.endswith(up_suffix):
                continue

            base = key[: -len(up_suffix)]
            down_key = f"{base}{down_suffix}"
            if down_key not in lora_sd:
                continue

            if base in pairs:
                existing = pairs[base]
                if existing["up_key"] != key or existing["down_key"] != down_key:
                    raise ValueError(f"ambiguous LoRA format for base '{base}'")
                continue

            up_tensor = lora_sd[key]
            down_tensor = lora_sd[down_key]
            if not isinstance(up_tensor, torch.Tensor) or not isinstance(down_tensor, torch.Tensor):
                raise ValueError(f"LoRA pair '{base}' has non-tensor up/down weights")
            if up_tensor.ndim != 2 or down_tensor.ndim != 2:
                raise ValueError(
                    f"LoRA pair '{base}' is not a standard 2D LoRA pair "
                    f"(got up.ndim={up_tensor.ndim}, down.ndim={down_tensor.ndim})"
                )

            alpha_key = f"{base}.alpha"
            dora_key = f"{base}.dora_scale"
            reshape_key = f"{base}.reshape_weight"
            mid_key = f"{base}{mid_suffix}" if mid_suffix is not None else None

            if dora_key in lora_sd:
                raise ValueError(
                    f"DoRA is not supported for offline merge without a base model reference: '{dora_key}'"
                )
            if reshape_key in lora_sd:
                raise ValueError(
                    f"reshape_weight is not supported by this merge node: '{reshape_key}'"
                )
            if mid_key is not None and mid_key in lora_sd:
                raise ValueError(
                    f"LoCon/LoRA mid weights are not supported by this merge node: '{mid_key}'"
                )

            pair_info = {
                "base": base,
                "up_key": key,
                "down_key": down_key,
                "alpha_key": alpha_key,
                "up": up_tensor,
                "down": down_tensor,
            }
            pairs[base] = pair_info
            recognized_tensor_keys.add(key)
            recognized_tensor_keys.add(down_key)
            if isinstance(lora_sd.get(alpha_key), torch.Tensor):
                recognized_tensor_keys.add(alpha_key)

    unsupported_tensor_keys = sorted(
        key
        for key, value in lora_sd.items()
        if isinstance(value, torch.Tensor) and key not in recognized_tensor_keys
    )
    if unsupported_tensor_keys:
        preview = ", ".join(unsupported_tensor_keys[:8])
        if len(unsupported_tensor_keys) > 8:
            preview += ", ..."
        raise ValueError(
            "merge node only supports standard LoRA up/down pairs right now; "
            f"unsupported tensor keys: {preview}"
        )

    return pairs


class LoadLoraOnly:
    """LoRA Load Only node.

    Load a LoRA without applying it. Use with other LoRA operation nodes.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "LoRA file name from the loras directory."},
                ),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"

    _NODE_NAME = "LoRA Load Only"
    DESCRIPTION = "Load a LoRA without applying it. Use with other LoRA operation nodes."

    def load_lora(self, lora_name: str):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        return (lora,)


class LoraLayersOperation:
    """LoRA Layers Operation node.

    Modify only selected LoRA layers by index.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The LoRA object to modify."}),
                "layer_pattern": (
                    "STRING",
                    {
                        "default": r".*transformer_blocks\.(\d+)\.",
                        "multiline": False,
                        "tooltip": "Regex used to detect a layer index. Group 1 should capture the index.",
                    },
                ),
                "layer_indices": (
                    "STRING",
                    {
                        "default": "59",
                        "multiline": False,
                        "tooltip": "Comma/range format, e.g. 0,1,2 or 10-15.",
                    },
                ),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Scale selected layers. Use 0 to zero-out selected layers.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LORA", "INT")
    RETURN_NAMES = ("modified_lora", "modified_keys_count")
    FUNCTION = "modify_lora"

    _NODE_NAME = "LoRA Layers Operation"
    DESCRIPTION = "Modify only selected LoRA layers by index."

    def modify_lora(
        self,
        lora: Dict[str, object],
        layer_pattern: str,
        layer_indices: str,
        scale_factor: float,
    ):
        try:
            selected_layers = set(_parse_layer_indices(layer_indices))
        except ValueError as exc:
            raise ValueError(f"failed to parse layer_indices '{layer_indices}': {exc}") from exc

        modified_lora = _clone_lora_state(lora)
        if not selected_layers:
            return (modified_lora, 0)

        pattern = re.compile(layer_pattern)
        modified_count = 0

        for key, value in modified_lora.items():
            if not isinstance(value, torch.Tensor):
                continue
            match = pattern.search(key)
            if not match:
                continue
            layer_id = _extract_layer_id(match)
            if layer_id is None or layer_id not in selected_layers:
                continue
            if scale_factor == 0:
                modified_lora[key] = torch.zeros_like(value)
            else:
                modified_lora[key] = value * scale_factor
            modified_count += 1

        print(
            f"[Yogurt LoRA] modified {modified_count} tensor keys in layers "
            f"{sorted(selected_layers)} with pattern '{layer_pattern}' and scale {scale_factor}"
        )
        return (modified_lora, modified_count)


class LoraScaleWeights:
    """LoRA Scale Weights node.

    Scale LoRA tensor weights globally so effect can be tuned while using strength=1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The LoRA object to scale."}),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Global multiplier for selected LoRA tensor keys.",
                    },
                ),
            },
            "optional": {
                "key_pattern": (
                    "STRING",
                    {
                        "default": r".*",
                        "multiline": False,
                        "tooltip": "Only keys matching this regex will be scaled.",
                    },
                ),
                "scale_alpha": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also scale .alpha tensor keys. Usually keep this off.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LORA", "INT")
    RETURN_NAMES = ("scaled_lora", "scaled_keys_count")
    FUNCTION = "scale_lora_weights"

    _NODE_NAME = "LoRA Scale Weights"
    DESCRIPTION = "Scale LoRA tensor weights globally so effect can be tuned while using strength=1."

    def scale_lora_weights(
        self,
        lora: Dict[str, object],
        scale_factor: float,
        key_pattern: str = r".*",
        scale_alpha: bool = False,
    ):
        modified = _clone_lora_state(lora)
        pattern = re.compile(key_pattern)
        changed = 0

        for key, value in modified.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.endswith(".alpha") and not scale_alpha:
                continue
            if not pattern.search(key):
                continue
            modified[key] = value * scale_factor
            changed += 1

        print(
            f"[Yogurt LoRA] globally scaled {changed} tensor keys "
            f"by {scale_factor} (pattern='{key_pattern}', scale_alpha={scale_alpha})"
        )
        return (modified, changed)


class LoraScaleAlpha:
    """LoRA Scale Alpha node.

    Scale only LoRA alpha metadata so the adjusted LoRA can be saved downstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": (
                    "LORA",
                    {"tooltip": "The LoRA object whose alpha values will be scaled."},
                ),
                "alpha_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Multiplier applied only to LoRA alpha/network_alpha keys.",
                    },
                ),
            },
            "optional": {
                "key_pattern": (
                    "STRING",
                    {
                        "default": r".*",
                        "multiline": False,
                        "tooltip": "Only alpha keys matching this regex will be scaled.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LORA", "INT")
    RETURN_NAMES = ("scaled_lora", "scaled_alpha_count")
    FUNCTION = "scale_lora_alpha"

    _NODE_NAME = "LoRA Scale Alpha"
    DESCRIPTION = "Scale only LoRA alpha metadata so the adjusted LoRA can be saved downstream."

    @staticmethod
    def _is_alpha_key(key: str) -> bool:
        return (
            key.endswith(".alpha")
            or key.endswith(".network_alpha")
            or key.endswith("_network_alpha")
        )

    def scale_lora_alpha(
        self,
        lora: Dict[str, object],
        alpha_scale: float,
        key_pattern: str = r".*",
    ):
        modified = _clone_lora_state(lora)
        pattern = re.compile(key_pattern)
        changed = 0

        for key, value in modified.items():
            if not self._is_alpha_key(key) or not pattern.search(key):
                continue
            if isinstance(value, torch.Tensor):
                modified[key] = value * alpha_scale
                changed += 1
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                modified[key] = torch.tensor(float(value) * alpha_scale)
                changed += 1

        for up_suffix, down_suffix, _mid_suffix in _LORA_PAIR_SUFFIXES:
            for key, up in list(modified.items()):
                if not key.endswith(up_suffix):
                    continue

                base = key[: -len(up_suffix)]
                alpha_key = f"{base}.alpha"
                if alpha_key in modified or not pattern.search(alpha_key):
                    continue

                down_key = f"{base}{down_suffix}"
                down = modified.get(down_key)
                if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
                    continue
                if down.ndim < 1:
                    continue

                rank = int(down.shape[0])
                modified[alpha_key] = _make_scalar_tensor(None, rank * alpha_scale, down)
                changed += 1

        print(
            f"[Yogurt LoRA] scaled {changed} alpha keys "
            f"by {alpha_scale} (pattern='{key_pattern}')"
        )
        return (modified, changed)


class MergeLoraToModel:
    """Merge LoRA To Model node.

    Apply loaded LoRA to model and optional CLIP.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to patch."}),
                "lora": ("LORA", {"tooltip": "LoRA object."}),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Strength for model patching.",
                    },
                ),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional CLIP to patch."}),
                "strength_clip": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Strength for CLIP patching.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_lora"

    _NODE_NAME = "Merge LoRA To Model"
    DESCRIPTION = "Apply loaded LoRA to model and optional CLIP."

    def apply_lora(
        self,
        model,
        lora: Dict[str, object],
        strength_model: float,
        clip=None,
        strength_clip: float = 1.0,
    ):
        if clip is None:
            strength_clip = 0.0

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


class LoraStatViewer:
    """LoRA Stat Viewer node.

    Inspect LoRA key patterns to help define regex and layer selection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "LoRA object to inspect."}),
            },
            "optional": {
                "show_all_keys": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When true, append all key names to output.",
                    },
                ),
                "sample_count": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "How many keys to show as samples.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_info",)
    FUNCTION = "view_lora_stats"

    _NODE_NAME = "LoRA Stat Viewer"
    DESCRIPTION = "Inspect LoRA key patterns to help define regex and layer selection."

    def view_lora_stats(
        self,
        lora: Dict[str, object],
        show_all_keys: bool = False,
        sample_count: int = 12,
    ):
        keys = list(lora.keys())
        lines: List[str] = []
        lines.append("=== Yogurt LoRA Statistics ===")
        lines.append(f"Total keys: {len(keys)}")

        key_types: Dict[str, int] = {}
        for key in keys:
            parts = key.split(".")
            kind = ".".join(parts[-3:]) if len(parts) >= 3 else key
            key_types[kind] = key_types.get(kind, 0) + 1

        lines.append("")
        lines.append("Key type histogram:")
        for key_type, count in sorted(key_types.items(), key=lambda item: (-item[1], item[0]))[:50]:
            lines.append(f"  - {key_type}: {count}")

        lines.append("")
        lines.append(f"Sample keys ({min(sample_count, len(keys))}):")
        for index, key in enumerate(keys[:sample_count]):
            lines.append(f"  [{index}] {key}")

        transformer_indices: Set[int] = set()
        for key in keys:
            for matched in re.findall(r"transformer_blocks\.(\d+)", key):
                transformer_indices.add(int(matched))
        if transformer_indices:
            sorted_indices = sorted(transformer_indices)
            preview = ", ".join(str(i) for i in sorted_indices[:20])
            suffix = " ..." if len(sorted_indices) > 20 else ""
            lines.append("")
            lines.append(f"transformer_blocks indices ({len(sorted_indices)}): {preview}{suffix}")

        if show_all_keys:
            lines.append("")
            lines.append("All keys:")
            for index, key in enumerate(keys):
                lines.append(f"  [{index}] {key}")

        return ("\n".join(lines),)


class CreateLoraMappingJson:
    """Create LoRA Mapping JSON node.

    Build a best-effort mapping from LoRA A keys to LoRA B keys.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loraA": ("LORA", {"tooltip": "Source LoRA key layout."}),
                "loraB": ("LORA", {"tooltip": "Target LoRA key layout."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mapping_json",)
    FUNCTION = "create_mapping_json"

    _NODE_NAME = "Create LoRA Mapping JSON"
    DESCRIPTION = "Build a best-effort mapping from LoRA A keys to LoRA B keys."

    @staticmethod
    def _extract_core_structure(key: str) -> str:
        match = re.search(r"transformer_blocks\.(\d+)", key)
        if match:
            return f"transformer_blocks.{match.group(1)}"

        match = re.search(r"(input|output|middle)_blocks\.(\d+)(\.(\d+))?", key)
        if match:
            block = match.group(1)
            index = match.group(2)
            sub = match.group(4)
            if sub is None:
                return f"{block}_blocks.{index}"
            return f"{block}_blocks.{index}.{sub}"

        simplified = re.sub(r"\.lora_(up|down|A|B)(\.weight|\.bias)?", "", key)
        simplified = re.sub(r"_lora(_(up|down|A|B))?(_weight|_bias)?", "", simplified)
        return simplified

    def create_mapping_json(self, loraA: Dict[str, object], loraB: Dict[str, object]):
        keys_a = list(loraA.keys())
        keys_b = list(loraB.keys())
        mapping: Dict[str, str] = {}

        for key in keys_a:
            if key in keys_b:
                mapping[key] = key

        grouped_a: Dict[str, List[str]] = {}
        grouped_b: Dict[str, List[str]] = {}

        for key in keys_a:
            if key in mapping:
                continue
            grouped_a.setdefault(self._extract_core_structure(key), []).append(key)

        for key in keys_b:
            grouped_b.setdefault(self._extract_core_structure(key), []).append(key)

        for structure, source_keys in grouped_a.items():
            target_keys = grouped_b.get(structure)
            if not target_keys:
                continue
            pair_count = min(len(source_keys), len(target_keys))
            for idx in range(pair_count):
                mapping[source_keys[idx]] = target_keys[idx]

        return (json.dumps(mapping, ensure_ascii=False, indent=2),)


class ConvertLoraKeys:
    """Convert LoRA Keys node.

    Rename LoRA keys by mapping JSON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "LoRA to convert."}),
                "mapping_json": (
                    "STRING",
                    {"tooltip": "JSON object: old_key -> new_key."},
                ),
            }
        }

    RETURN_TYPES = ("LORA", "INT")
    RETURN_NAMES = ("converted_lora", "converted_count")
    FUNCTION = "convert_lora"

    _NODE_NAME = "Convert LoRA Keys"
    DESCRIPTION = "Rename LoRA keys by mapping JSON."

    def convert_lora(self, lora: Dict[str, object], mapping_json: str):
        mapping = json.loads(mapping_json)
        if not isinstance(mapping, dict):
            raise ValueError("mapping_json must decode to an object")

        converted: Dict[str, object] = {}
        converted_count = 0
        for old_key, new_key in mapping.items():
            if old_key not in lora:
                continue
            value = lora[old_key]
            converted[str(new_key)] = value.clone() if isinstance(value, torch.Tensor) else value
            converted_count += 1

        print(f"[Yogurt LoRA] converted {converted_count} keys using mapping json")
        return (converted, converted_count)


class LoraSimpleAdd:
    """LoRA Simple Add node.

    Simple weighted sum of two LoRA states.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loraA": ("LORA", {"tooltip": "Base LoRA."}),
                "loraB": ("LORA", {"tooltip": "Secondary LoRA."}),
                "alpha_a": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "alpha_b": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("combined_lora",)
    FUNCTION = "simple_add_lora"

    _NODE_NAME = "LoRA Simple Add"
    DESCRIPTION = "Simple weighted sum of two LoRA states."

    def simple_add_lora(
        self,
        loraA: Dict[str, object],
        loraB: Dict[str, object],
        alpha_a: float = 1.0,
        alpha_b: float = 1.0,
    ):
        combined: Dict[str, object] = {}
        for key, value in loraA.items():
            if isinstance(value, torch.Tensor):
                combined[key] = alpha_a * value.clone()
            else:
                combined[key] = value

        for key, value in loraB.items():
            if not isinstance(value, torch.Tensor):
                if key not in combined:
                    combined[key] = value
                continue

            if key in combined and isinstance(combined[key], torch.Tensor):
                combined[key] = combined[key] + alpha_b * value
            else:
                combined[key] = alpha_b * value
        return (combined,)


class LoraAdd:
    """LoRA Add (Rank Aware) node.

    Merge two LoRAs, with SVD rank alignment when ranks differ.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loraA": ("LORA", {"tooltip": "Base LoRA."}),
                "loraB": ("LORA", {"tooltip": "Second LoRA."}),
                "alpha_a": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "alpha_b": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "target_rank": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Use -1 for auto(min(rankA, rankB)).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("merged_lora",)
    FUNCTION = "add_lora"

    _NODE_NAME = "LoRA Add (Rank Aware)"
    DESCRIPTION = "Merge two LoRAs, with SVD rank alignment when ranks differ."

    @staticmethod
    def _extract_alpha(lora_sd: Dict[str, object], down_key: str) -> float:
        base = re.sub(r"\.lora_down\.weight$", "", down_key)
        alpha_key = f"{base}.alpha"
        alpha = lora_sd.get(alpha_key)
        if alpha is None:
            return float(lora_sd[down_key].shape[0])
        if isinstance(alpha, torch.Tensor):
            return float(alpha.item())
        return float(alpha)

    def _absorb_alpha(self, lora_sd: Dict[str, object]) -> Dict[str, object]:
        down_keys = [k for k in lora_sd if k.endswith(".weight") and ".lora_down." in k]
        for down_key in down_keys:
            up_key = down_key.replace(".lora_down.", ".lora_up.")
            if up_key not in lora_sd:
                continue
            down_tensor = lora_sd.get(down_key)
            if not isinstance(down_tensor, torch.Tensor):
                continue
            rank = down_tensor.shape[0]
            if rank <= 0:
                continue
            scale = self._extract_alpha(lora_sd, down_key) / float(rank)
            lora_sd[down_key] = down_tensor * scale
        return lora_sd

    @staticmethod
    def _svd_align_rank(
        lora_down: torch.Tensor, lora_up: torch.Tensor, target_rank: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank, input_dim = lora_down.shape
        output_dim, _ = lora_up.shape
        k = min(target_rank, rank, input_dim, output_dim)
        if k == rank:
            return lora_down, lora_up

        delta_w = lora_up @ lora_down
        u, s, vt = torch.linalg.svd(delta_w, full_matrices=False)
        u_k = u[:, :k]
        s_k = s[:k]
        vt_k = vt[:k, :]
        sqrt_s = torch.sqrt(s_k).unsqueeze(1)
        new_up = u_k * sqrt_s.T
        new_down = sqrt_s * vt_k
        return new_down, new_up

    def add_lora(
        self,
        loraA: Dict[str, object],
        loraB: Dict[str, object],
        alpha_a: float = 1.0,
        alpha_b: float = 1.0,
        target_rank: int = -1,
    ):
        lora_a = self._absorb_alpha(_clone_lora_state(loraA))
        lora_b = self._absorb_alpha(_clone_lora_state(loraB))

        down_keys_a = {k for k in lora_a if k.endswith(".weight") and ".lora_down." in k}
        down_keys_b = {k for k in lora_b if k.endswith(".weight") and ".lora_down." in k}
        common = down_keys_a & down_keys_b

        for down_key in common:
            up_key = down_key.replace(".lora_down.", ".lora_up.")
            if up_key not in lora_a or up_key not in lora_b:
                continue

            down_a = lora_a[down_key]
            up_a = lora_a[up_key]
            down_b = lora_b[down_key]
            up_b = lora_b[up_key]

            if not all(isinstance(x, torch.Tensor) for x in [down_a, up_a, down_b, up_b]):
                continue

            rank_a = down_a.shape[0]
            rank_b = down_b.shape[0]
            merged_rank = target_rank if target_rank != -1 else min(rank_a, rank_b)

            if rank_a != merged_rank:
                down_a, up_a = self._svd_align_rank(down_a, up_a, merged_rank)
            if rank_b != merged_rank:
                down_b, up_b = self._svd_align_rank(down_b, up_b, merged_rank)

            lora_a[down_key] = alpha_a * down_a + alpha_b * down_b
            lora_a[up_key] = alpha_a * up_a + alpha_b * up_b

            alpha_key = re.sub(r"\.lora_down\.weight$", ".alpha", down_key)
            if alpha_key in loraA or alpha_key in loraB:
                lora_a[alpha_key] = torch.tensor(float(merged_rank))

        for key, value in lora_b.items():
            if key not in lora_a:
                lora_a[key] = value

        return (lora_a,)


class LoraMerge:
    """LoRA Merge Full Rank node.

    Merge up to five standard LoRAs exactly by concatenating rank dimensions. Fast and preserves the summed model-side effect exactly, but output rank/file size grow. Does not support DoRA or LoCon/reshape variants.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora1": ("LORA", {"tooltip": "First LoRA."}),
                "strength1": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "lora2": ("LORA", {"tooltip": "Second LoRA."}),
                "strength2": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "lora3": ("LORA", {"tooltip": "Third LoRA."}),
                "strength3": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "lora4": ("LORA", {"tooltip": "Fourth LoRA."}),
                "strength4": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "lora5": ("LORA", {"tooltip": "Fifth LoRA."}),
                "strength5": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("LORA", "INT", "STRING")
    RETURN_NAMES = ("merged_lora", "merged_pairs_count", "report")
    FUNCTION = "merge_lora_full_rank"

    _NODE_NAME = "LoRA Merge Full Rank"
    DESCRIPTION = (
        "Merge up to five standard LoRAs exactly by concatenating rank dimensions. "
        "Fast and preserves the summed model-side effect exactly, but output rank/file "
        "size grow. Does not support DoRA or LoCon/reshape variants."
    )

    @staticmethod
    def _copy_scaled_single_pair(
        merged: Dict[str, object],
        pair: Dict[str, object],
        lora_sd: Dict[str, object],
        strength: float,
    ) -> bool:
        if strength == 0:
            return False

        down = pair["down"]
        up = pair["up"]
        if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
            raise ValueError(f"invalid LoRA pair '{pair['base']}'")

        rank = int(down.shape[0])
        alpha = _extract_lora_alpha(lora_sd, str(pair["alpha_key"]), down)
        scale = strength * (alpha / float(rank))
        base = str(pair["base"])

        merged[f"{base}.lora_down.weight"] = (down.clone() * scale).contiguous()
        merged[f"{base}.lora_up.weight"] = up.clone().contiguous()
        merged[f"{base}.alpha"] = _make_rank_tensor(lora_sd.get(str(pair["alpha_key"])), rank, down)
        return True

    @staticmethod
    def _concat_pairs_for_base(base: str, sources: List[tuple[Dict[str, object], Dict[str, object], float]]):
        first_sd, first_pair, _ = sources[0]
        ref_down = first_pair["down"]
        ref_up = first_pair["up"]
        if not isinstance(ref_down, torch.Tensor) or not isinstance(ref_up, torch.Tensor):
            raise ValueError(f"invalid LoRA pair '{base}'")

        up_parts: List[torch.Tensor] = []
        down_parts: List[torch.Tensor] = []
        total_rank = 0

        for lora_sd, pair, strength in sources:
            down = pair["down"]
            up = pair["up"]
            if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
                raise ValueError(f"invalid LoRA pair '{base}'")
            if up.shape[0] != ref_up.shape[0] or down.shape[1] != ref_down.shape[1]:
                raise ValueError(
                    f"shape mismatch while merging '{base}': "
                    f"expected up[0]={ref_up.shape[0]}, down[1]={ref_down.shape[1]}, "
                    f"got up={tuple(up.shape)}, down={tuple(down.shape)}"
                )

            rank = int(down.shape[0])
            alpha = _extract_lora_alpha(lora_sd, str(pair["alpha_key"]), down)
            scale = strength * (alpha / float(rank))
            up_parts.append(up.to(dtype=ref_up.dtype, device=ref_up.device))
            down_parts.append((down.to(dtype=ref_down.dtype, device=ref_down.device) * scale))
            total_rank += rank

        merged_up = torch.cat(up_parts, dim=1).contiguous()
        merged_down = torch.cat(down_parts, dim=0).contiguous()
        alpha_tensor = _make_rank_tensor(first_sd.get(str(first_pair["alpha_key"])), total_rank, ref_down)
        return merged_up, merged_down, alpha_tensor, total_rank

    def merge_lora_full_rank(
        self,
        lora1: Dict[str, object],
        strength1: float = 1.0,
        lora2: Optional[Dict[str, object]] = None,
        strength2: float = 1.0,
        lora3: Optional[Dict[str, object]] = None,
        strength3: float = 1.0,
        lora4: Optional[Dict[str, object]] = None,
        strength4: float = 1.0,
        lora5: Optional[Dict[str, object]] = None,
        strength5: float = 1.0,
    ):
        raw_sources = [
            (lora1, strength1),
            (lora2, strength2),
            (lora3, strength3),
            (lora4, strength4),
            (lora5, strength5),
        ]
        prepared_sources: List[tuple[Dict[str, object], Dict[str, Dict[str, object]], float]] = []
        for lora, strength in raw_sources:
            if lora is None:
                continue
            converted = comfy.lora_convert.convert_lora(_clone_lora_state(lora))
            pairs = _scan_mergeable_lora_pairs(converted)
            prepared_sources.append((converted, pairs, strength))

        if not prepared_sources:
            raise ValueError("at least one LoRA is required")

        all_bases = sorted(
            {
                base
                for _, pairs, _ in prepared_sources
                for base in pairs.keys()
            }
        )
        if not all_bases:
            raise ValueError("no mergeable standard LoRA pairs found")

        merged: Dict[str, object] = {}
        merged_count = 0
        progress = comfy.utils.ProgressBar(len(all_bases))
        progress.update_absolute(0)
        report_lines: List[str] = []
        report_lines.append("=== LoRA Merge Full Rank Report ===")
        report_lines.append(f"source_loras={len(prepared_sources)}")

        for index, base in enumerate(all_bases, start=1):
            base_sources: List[tuple[Dict[str, object], Dict[str, object], float]] = []
            for lora_sd, pairs, strength in prepared_sources:
                pair = pairs.get(base)
                if pair is None or strength == 0:
                    continue
                base_sources.append((lora_sd, pair, strength))

            if not base_sources:
                report_lines.append(f"- {base}: skipped because all strengths are 0")
                progress.update_absolute(index)
                continue

            if len(base_sources) == 1:
                lora_sd, pair, strength = base_sources[0]
                copied = self._copy_scaled_single_pair(merged, pair, lora_sd, strength)
                if copied:
                    merged_count += 1
                    report_lines.append(
                        f"- {base}: copied single pair with strength {strength}"
                    )
                else:
                    report_lines.append(f"- {base}: skipped single pair because strength is 0")
                progress.update_absolute(index)
                continue

            merged_up, merged_down, alpha_tensor, merged_rank = self._concat_pairs_for_base(base, base_sources)
            merged[f"{base}.lora_up.weight"] = merged_up
            merged[f"{base}.lora_down.weight"] = merged_down
            merged[f"{base}.alpha"] = alpha_tensor
            merged_count += 1
            report_lines.append(
                f"- {base}: merged {len(base_sources)} LoRAs -> rank {merged_rank}"
            )
            progress.update_absolute(index)

        report_lines.append(f"merged_pairs={merged_count}")
        report = "\n".join(report_lines)
        print(f"[Yogurt LoRA] full-rank merged pairs: {merged_count}/{len(all_bases)}")
        return (merged, merged_count, report)


class LoraRankCompress:
    """LoRA Compress node.

    Compress LoRA rank with SVD for standard .lora_down/.lora_up pairs. Optionally absorb alpha/rank first to preserve the actual LoRA effect before compression.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "LoRA to compress."}),
                "target_rank": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Target rank. Use -1 to auto-pick by energy_keep_ratio.",
                    },
                ),
                "energy_keep_ratio": (
                    "FLOAT",
                    {
                        "default": 0.98,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Only used when target_rank=-1. Keeps this fraction of SVD energy.",
                    },
                ),
                "min_rank": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Minimum rank when auto-selecting.",
                    },
                ),
                "key_pattern": (
                    "STRING",
                    {
                        "default": r".*",
                        "multiline": False,
                        "tooltip": "Regex filter for down-key names.",
                    },
                ),
                "update_alpha": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Rewrite alpha after compression so saved LoRA keeps the expected scale.",
                    },
                ),
                "absorb_alpha": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When true, reconstruct full LoRA effect using alpha/rank before SVD compression.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LORA", "INT", "STRING")
    RETURN_NAMES = ("compressed_lora", "compressed_layers_count", "report")
    FUNCTION = "compress_rank"

    _NODE_NAME = "LoRA Compress"
    DESCRIPTION = (
        "Compress LoRA rank with SVD for standard .lora_down/.lora_up pairs. "
        "Optionally absorb alpha/rank first to preserve the actual LoRA effect before compression."
    )

    @staticmethod
    def _pick_rank_from_energy(
        singular_values: torch.Tensor, energy_keep_ratio: float, min_rank: int, max_rank: int
    ) -> int:
        if singular_values.numel() == 0:
            return min_rank
        if energy_keep_ratio <= 0:
            return min_rank
        if energy_keep_ratio >= 1:
            return max_rank

        energy = torch.cumsum(singular_values * singular_values, dim=0)
        total = energy[-1]
        if total <= 0:
            return min_rank

        threshold = total * energy_keep_ratio
        index = int(torch.searchsorted(energy, threshold, right=False).item())
        rank = index + 1
        return max(min_rank, min(rank, max_rank))

    @staticmethod
    def _make_rank_tensor(value: object, rank: int, ref: torch.Tensor) -> torch.Tensor:
        return _make_rank_tensor(value, rank, ref)

    @staticmethod
    def _reconstruct_delta(
        up: torch.Tensor,
        down: torch.Tensor,
        alpha: float,
        absorb_alpha: bool,
    ) -> torch.Tensor:
        up_f32 = up.to(dtype=torch.float32)
        down_f32 = down.to(dtype=torch.float32)
        if absorb_alpha:
            scale = alpha / float(down.shape[0])
            return up_f32 @ (down_f32 * scale)
        return up_f32 @ down_f32

    def compress_rank(
        self,
        lora: Dict[str, object],
        target_rank: int = -1,
        energy_keep_ratio: float = 0.98,
        min_rank: int = 1,
        key_pattern: str = r".*",
        update_alpha: bool = True,
        absorb_alpha: bool = True,
    ):
        if target_rank != -1 and target_rank < 1:
            raise ValueError("target_rank must be -1 or >= 1")
        if min_rank < 1:
            raise ValueError("min_rank must be >= 1")

        pattern = re.compile(key_pattern)
        converted = comfy.lora_convert.convert_lora(_clone_lora_state(lora))
        pairs = _scan_mergeable_lora_pairs(converted)
        compressed: Dict[str, object] = {}

        changed = 0
        scanned_pairs = 0
        report_lines: List[str] = []
        report_lines.append("=== LoRA Rank Compress Report ===")
        report_lines.append(
            f"mode={'target_rank' if target_rank != -1 else 'auto_energy'} | "
            f"target_rank={target_rank} | energy_keep_ratio={energy_keep_ratio} | "
            f"min_rank={min_rank} | pattern={key_pattern} | absorb_alpha={absorb_alpha}"
        )

        all_bases = sorted(base for base in pairs.keys() if pattern.search(f"{base}.lora_down.weight"))
        progress = comfy.utils.ProgressBar(len(all_bases))
        progress.update_absolute(0)

        for index, base in enumerate(all_bases, start=1):
            pair = pairs[base]
            down_key = f"{base}.lora_down.weight"
            up_key = f"{base}.lora_up.weight"
            alpha_key = f"{base}.alpha"
            down = pair["down"]
            up = pair["up"]
            if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
                progress.update_absolute(index)
                continue
            if down.ndim != 2 or up.ndim != 2:
                progress.update_absolute(index)
                continue

            scanned_pairs += 1
            original_rank = int(down.shape[0])
            max_rank = int(min(original_rank, down.shape[1], up.shape[0]))
            if max_rank < 1:
                progress.update_absolute(index)
                continue

            alpha = _extract_lora_alpha(converted, alpha_key, down)
            # SVD on reconstructed deltaW in float32 for stability.
            delta_w = self._reconstruct_delta(up, down, alpha, absorb_alpha)
            u, s, vt = torch.linalg.svd(delta_w, full_matrices=False)

            if target_rank == -1:
                rank = self._pick_rank_from_energy(
                    singular_values=s,
                    energy_keep_ratio=energy_keep_ratio,
                    min_rank=min_rank,
                    max_rank=max_rank,
                )
            else:
                rank = max(min_rank, min(int(target_rank), max_rank))

            if rank >= original_rank:
                compressed[down_key] = down.clone().contiguous()
                compressed[up_key] = up.clone().contiguous()
                if alpha_key in converted:
                    alpha_value = converted[alpha_key]
                    compressed[alpha_key] = (
                        alpha_value.clone() if isinstance(alpha_value, torch.Tensor) else alpha_value
                    )
                progress.update_absolute(index)
                continue

            sqrt_s = torch.sqrt(s[:rank]).unsqueeze(1)
            up_new = (
                u[:, :rank] * sqrt_s.T
            ).to(dtype=up.dtype, device=up.device).contiguous()
            down_new = (
                sqrt_s * vt[:rank, :]
            ).to(dtype=down.dtype, device=down.device).contiguous()

            compressed[down_key] = down_new
            compressed[up_key] = up_new

            if update_alpha or absorb_alpha:
                if absorb_alpha:
                    alpha_out = float(rank)
                else:
                    alpha_out = float(rank) * float(alpha) / float(original_rank)
                compressed[alpha_key] = _make_scalar_tensor(
                    converted.get(alpha_key), alpha_out, down_new
                )
            elif alpha_key in converted:
                alpha_value = converted[alpha_key]
                compressed[alpha_key] = (
                    alpha_value.clone() if isinstance(alpha_value, torch.Tensor) else alpha_value
                )

            changed += 1
            report_lines.append(
                f"- {down_key}: {original_rank} -> {rank} "
                f"(alpha={alpha:.4f}, absorb_alpha={absorb_alpha})"
            )
            progress.update_absolute(index)

        report_lines.append(f"scanned_pairs={scanned_pairs}")
        report_lines.append(f"compressed_pairs={changed}")
        report = "\n".join(report_lines)
        print(f"[Yogurt LoRA] rank compressed pairs: {changed}/{scanned_pairs}")

        return (compressed, changed, report)


class SaveLora:
    """Save LoRA node.

    Save LoRA state as safetensors.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "LoRA state to save."}),
                "filename": (
                    "STRING",
                    {
                        "default": "yogurt_lora.safetensors",
                        "tooltip": "Target filename (extension optional).",
                    },
                ),
            },
            "optional": {
                "output_dir": (
                    "STRING",
                    {
                        "default": folder_paths.get_output_directory(),
                        "tooltip": "Directory path. Defaults to ComfyUI output directory.",
                    },
                ),
                "remove_zero_layers": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Drop tensor keys that are all zeros before saving.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora"
    OUTPUT_NODE = True

    _NODE_NAME = "Save LoRA"
    DESCRIPTION = "Save LoRA state as safetensors."

    @staticmethod
    def _remove_zero_layers(lora: Dict[str, object]) -> Dict[str, object]:
        filtered: Dict[str, object] = {}
        for key, value in lora.items():
            if isinstance(value, torch.Tensor):
                if torch.count_nonzero(value).item() != 0:
                    filtered[key] = value
            else:
                filtered[key] = value
        return filtered

    @staticmethod
    def _make_tensors_contiguous(lora: Dict[str, object]) -> Dict[str, object]:
        packed: Dict[str, object] = {}
        for key, value in lora.items():
            if isinstance(value, torch.Tensor):
                packed[key] = value.contiguous().clone()
            else:
                packed[key] = value
        return packed

    def save_lora(
        self,
        lora: Dict[str, object],
        filename: str,
        output_dir: Optional[str] = None,
        remove_zero_layers: bool = False,
    ):
        target_dir = output_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        output_path = os.path.join(target_dir, filename)
        if not output_path.lower().endswith(".safetensors"):
            output_path = f"{output_path}.safetensors"

        filtered_lora = self._remove_zero_layers(lora) if remove_zero_layers else lora
        packed_lora = self._make_tensors_contiguous(filtered_lora)
        save_file(packed_lora, output_path)
        print(
            f"[Yogurt LoRA] saved to: {output_path} "
            f"({len(lora)} -> {len(filtered_lora)} keys)"
        )
        return {}
