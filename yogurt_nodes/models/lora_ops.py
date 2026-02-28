import json
import os
import re
from typing import Dict, List, Optional, Set

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


class LoadLoraOnly:
    """
    Load a LoRA file into a reusable LORA object.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
    """
    Scale or zero selected layers in a LoRA by regex + index selection.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
    """
    Globally scale LoRA tensor weights so downstream strength=1 has custom intensity.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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


class MergeLoraToModel:
    """
    Apply a preloaded LoRA object to model and optional CLIP.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
    """
    Summarize LoRA key distribution and print sample keys.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
    """
    Create a key mapping JSON to convert one LoRA key layout to another.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
    """
    Convert LoRA key names based on JSON mapping text.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
    """
    Add two LoRA state dicts key-by-key with scalar weights.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

    def simple_add_lora(
        self,
        loraA: Dict[str, object],
        loraB: Dict[str, object],
        alpha_a: float = 1.0,
        alpha_b: float = 1.0,
    ):
        combined = _clone_lora_state(loraA)
        for key, value in loraB.items():
            if not isinstance(value, torch.Tensor):
                if key not in combined:
                    combined[key] = value
                continue

            if key in combined and isinstance(combined[key], torch.Tensor):
                combined[key] = alpha_a * combined[key] + alpha_b * value
            else:
                combined[key] = alpha_b * value
        return (combined,)


class LoraAdd:
    """
    Rank-aware LoRA merge using SVD alignment for mismatched ranks.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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


class LoraRankCompress:
    """
    Compress standard LoRA down/up pairs with SVD rank reduction.
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
                        "tooltip": "When .alpha exists, set it to compressed rank.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LORA", "INT", "STRING")
    RETURN_NAMES = ("compressed_lora", "compressed_layers_count", "report")
    FUNCTION = "compress_rank"

    _NODE_NAME = "LoRA Rank Compress (SVD)"
    DESCRIPTION = (
        "Compress LoRA rank with SVD for standard .lora_down/.lora_up pairs. "
        "Useful for reducing size and making strength behavior easier to control."
    )
    CATEGORY = "YogurtNodes/Models/LoRA"

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
        if isinstance(value, torch.Tensor):
            return torch.tensor(float(rank), dtype=value.dtype, device=value.device)
        return torch.tensor(float(rank), dtype=ref.dtype, device=ref.device)

    def compress_rank(
        self,
        lora: Dict[str, object],
        target_rank: int = -1,
        energy_keep_ratio: float = 0.98,
        min_rank: int = 1,
        key_pattern: str = r".*",
        update_alpha: bool = True,
    ):
        if target_rank != -1 and target_rank < 1:
            raise ValueError("target_rank must be -1 or >= 1")
        if min_rank < 1:
            raise ValueError("min_rank must be >= 1")

        pattern = re.compile(key_pattern)
        compressed = _clone_lora_state(lora)

        changed = 0
        scanned_pairs = 0
        report_lines: List[str] = []
        report_lines.append("=== LoRA Rank Compress Report ===")
        report_lines.append(
            f"mode={'target_rank' if target_rank != -1 else 'auto_energy'} | "
            f"target_rank={target_rank} | energy_keep_ratio={energy_keep_ratio} | "
            f"min_rank={min_rank} | pattern={key_pattern}"
        )

        down_keys = [
            key
            for key in compressed
            if key.endswith(".weight") and ".lora_down." in key and pattern.search(key)
        ]
        down_keys.sort()

        for down_key in down_keys:
            up_key = down_key.replace(".lora_down.", ".lora_up.")
            if up_key not in compressed:
                continue

            down = compressed[down_key]
            up = compressed[up_key]
            if not isinstance(down, torch.Tensor) or not isinstance(up, torch.Tensor):
                continue
            if down.ndim != 2 or up.ndim != 2:
                continue

            scanned_pairs += 1
            original_rank = int(down.shape[0])
            max_rank = int(min(original_rank, down.shape[1], up.shape[0]))
            if max_rank < 1:
                continue

            # SVD on deltaW in float32 for stability and broad backend compatibility.
            delta_w = up.to(dtype=torch.float32) @ down.to(dtype=torch.float32)
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
                continue

            sqrt_s = torch.sqrt(s[:rank]).unsqueeze(1)
            up_new = (u[:, :rank] * sqrt_s.T).to(dtype=up.dtype, device=up.device)
            down_new = (sqrt_s * vt[:rank, :]).to(dtype=down.dtype, device=down.device)

            compressed[down_key] = down_new
            compressed[up_key] = up_new

            if update_alpha:
                alpha_key = re.sub(r"\.lora_down\.weight$", ".alpha", down_key)
                if alpha_key in compressed:
                    compressed[alpha_key] = self._make_rank_tensor(
                        compressed[alpha_key], rank, down_new
                    )

            changed += 1
            report_lines.append(f"- {down_key}: {original_rank} -> {rank}")

        report_lines.append(f"scanned_pairs={scanned_pairs}")
        report_lines.append(f"compressed_pairs={changed}")
        report = "\n".join(report_lines)
        print(f"[Yogurt LoRA] rank compressed pairs: {changed}/{scanned_pairs}")

        return (compressed, changed, report)


class SaveLora:
    """
    Save LoRA state to output directory as safetensors.
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
    CATEGORY = "YogurtNodes/Models/LoRA"

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
        save_file(filtered_lora, output_path)
        print(
            f"[Yogurt LoRA] saved to: {output_path} "
            f"({len(lora)} -> {len(filtered_lora)} keys)"
        )
        return {}
