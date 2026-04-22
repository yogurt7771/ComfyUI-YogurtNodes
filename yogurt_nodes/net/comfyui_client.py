from typing import List

import numpy as np
import torch
from PIL import Image

from ..utils import ANY_TYPE, ComfyUIClient


def _parse_nodes(nodes: str) -> List[str]:
    """
    Split comma/newline separated node names into a clean list.
    """
    items = []
    for part in nodes.replace("\r", "\n").split("\n"):
        for piece in part.split(","):
            name = piece.strip()
            if name:
                items.append(name)
    return items


def _tensor_to_pil(image_data) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor (N, H, W, C) to a PIL image.
    """
    if isinstance(image_data, Image.Image):
        return image_data

    if not isinstance(image_data, torch.Tensor):
        raise TypeError("image must be a torch tensor or PIL image")

    tensor = image_data
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor * 255).to(torch.uint8).numpy()

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)

    return Image.fromarray(array)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    tensor = torch.from_numpy(array)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


class ComfyUIClientLoad:
    """ComfyUI Client Load node.

    配置 ComfyUI 客户端实例，供后续节点复用
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8188",
                        "multiline": False,
                        "tooltip": "ComfyUI 服务器地址",
                    },
                ),
                "workflow": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "{}",
                        "tooltip": "ComfyUI 工作流 JSON",
                    },
                ),
            },
            "optional": {
                "poll_interval": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.05,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "轮询 history 的间隔（秒）",
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 1.0,
                        "tooltip": "等待结果的超时时长（秒），0 表示不超时",
                    },
                ),
                "auto_connect": (
                    ["true", "false"],
                    {"default": "true", "tooltip": "是否立即建立 HTTP 会话（上传图片时需要）"},
                ),
            },
        }

    RETURN_TYPES = ("COMFYUI_CLIENT",)
    RETURN_NAMES = ("client",)

    FUNCTION = "load"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Load"
    DESCRIPTION = "配置 ComfyUI 客户端实例，供后续节点复用"
    CATEGORY = "YogurtNodes/Net"

    def load(
        self,
        server: str,
        workflow: str,
        poll_interval: float = 0.2,
        timeout: float = 120.0,
        auto_connect: str = "true",
    ):
        client = ComfyUIClient(
            server,
            workflow,
            poll_interval=poll_interval,
            timeout=None if timeout <= 0 else timeout,
        )
        if auto_connect == "true":
            client.connect()
        return (client,)


class ComfyUIClientSetString:
    """ComfyUI Client Set String node.

    向工作流节点输入设置字符串
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("COMFYUI_CLIENT",),
                "node": ("STRING", {"tooltip": "节点 ID 或节点名"}),
                "key": ("STRING", {"tooltip": "输入口名称"}),
                "value": ("STRING", {"multiline": True, "tooltip": "要设置的字符串值"}),
            },
        }

    RETURN_TYPES = ("COMFYUI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "set_value"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Set String"
    DESCRIPTION = "向工作流节点输入设置字符串"
    CATEGORY = "YogurtNodes/Net"

    def set_value(self, client: ComfyUIClient, node: str, key: str, value: str):
        client.set_data(node, key, value)
        return (client,)


class ComfyUIClientSetFloat:
    """ComfyUI Client Set Float node.

    向工作流节点输入设置浮点数
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("COMFYUI_CLIENT",),
                "node": ("STRING", {"tooltip": "节点 ID 或节点名"}),
                "key": ("STRING", {"tooltip": "输入口名称"}),
                "value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "要设置的浮点值",
                    },
                ),
            },
        }

    RETURN_TYPES = ("COMFYUI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "set_value"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Set Float"
    DESCRIPTION = "向工作流节点输入设置浮点数"
    CATEGORY = "YogurtNodes/Net"

    def set_value(self, client: ComfyUIClient, node: str, key: str, value: float):
        client.set_data(node, key, float(value))
        return (client,)


class ComfyUIClientSetInt:
    """ComfyUI Client Set Int node.

    向工作流节点输入设置整数
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("COMFYUI_CLIENT",),
                "node": ("STRING", {"tooltip": "节点 ID 或节点名"}),
                "key": ("STRING", {"tooltip": "输入口名称"}),
                "value": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "要设置的整数值",
                    },
                ),
            },
        }

    RETURN_TYPES = ("COMFYUI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "set_value"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Set Int"
    DESCRIPTION = "向工作流节点输入设置整数"
    CATEGORY = "YogurtNodes/Net"

    def set_value(self, client: ComfyUIClient, node: str, key: str, value: int):
        client.set_data(node, key, int(value))
        return (client,)


class ComfyUIClientSetSeed:
    """ComfyUI Client Set Seed node.

    为工作流中的节点设置随机种子
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("COMFYUI_CLIENT",),
                "seed": ("INT", {"default": 0, "tooltip": "统一设置的随机种子"}),
            },
            "optional": {
                "nodes": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "指定节点（可用逗号/换行分隔 ID 或标题），留空则应用到全部节点",
                    },
                )
            },
        }

    RETURN_TYPES = ("COMFYUI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "set_seed"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Set Seed"
    DESCRIPTION = "为工作流中的节点设置随机种子"
    CATEGORY = "YogurtNodes/Net"

    def set_seed(self, client: ComfyUIClient, seed: int, nodes: str = ""):
        target_nodes = _parse_nodes(nodes)
        client.set_seed(int(seed), target_nodes or None)
        return (client,)


class ComfyUIClientSetImage:
    """ComfyUI Client Set Image node.

    上传图片并写入工作流节点输入
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("COMFYUI_CLIENT",),
                "node": ("STRING", {"tooltip": "节点 ID 或节点名"}),
                "key": ("STRING", {"tooltip": "输入口名称"}),
                "image": ("IMAGE", {"tooltip": "要上传的图片（取第一张）"}),
            },
        }

    RETURN_TYPES = ("COMFYUI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "set_image"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Set Image"
    DESCRIPTION = "上传图片并写入工作流节点输入"
    CATEGORY = "YogurtNodes/Net"

    def set_image(self, client: ComfyUIClient, node: str, key: str, image):
        pil_image = _tensor_to_pil(image)
        client.set_data(node, key, pil_image)
        return (client,)


class ComfyUIClientRun:
    """ComfyUI Client Run node.

    提交工作流并等待结果返回
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("COMFYUI_CLIENT",),
            },
            "optional": {
                "nodes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "仅收集这些节点输出（逗号/换行分隔），留空获取全部",
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 1.0,
                        "tooltip": "覆盖客户端超时时间（秒），0 表示沿用客户端设置",
                    },
                ),
                "poll_interval": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "覆盖客户端轮询间隔，0 表示沿用客户端设置",
                    },
                ),
            },
        }

    RETURN_TYPES = ("COMFYUI_RESULTS", "STRING")
    RETURN_NAMES = ("results", "prompt_id")
    FUNCTION = "run"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Run"
    DESCRIPTION = "提交工作流并等待结果返回"
    CATEGORY = "YogurtNodes/Net"

    def run(
        self,
        client: ComfyUIClient,
        nodes: str = "",
        timeout: float = 0.0,
        poll_interval: float = 0.0,
    ):
        node_list = _parse_nodes(nodes)
        actual_timeout = None if timeout <= 0 else timeout
        poll = None if poll_interval <= 0 else poll_interval
        result = client.run(
            node_names=node_list or None,
            timeout=actual_timeout,
            poll_interval=poll,
        )
        return (result, result.get("prompt_id", ""))


class ComfyUIClientGetOutput:
    """ComfyUI Client Get Output node.

    根据节点 ID/名称，从结果包中取出该节点的全部输出列表
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("COMFYUI_RESULTS",),
                "node": (
                    "STRING",
                    {"default": "", "tooltip": "节点 ID 或标题；留空且仅有单一节点输出时自动取用"},
                ),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "get_output"
    OUTPUT_NODE = False

    _NODE_NAME = "ComfyUI Client Get Output"
    DESCRIPTION = "根据节点 ID/名称，从结果包中取出该节点的全部输出列表"
    CATEGORY = "YogurtNodes/Net"

    def _resolve_node_id(self, results: dict, node: str, outputs: dict) -> str:
        target = node.strip()
        if target:
            if target in outputs:
                return target
            titles = results.get("node_titles", {})
            if target in titles:
                return titles[target]
            all_outputs = results.get("all_outputs", {})
            if target in all_outputs:
                return target
            raise RuntimeError(f"未找到节点：{target}")

        if len(outputs) == 1:
            return next(iter(outputs.keys()))

        raise RuntimeError("请指定节点 ID 或名称")

    def _convert_entry(self, entry):
        if isinstance(entry, dict) and entry:
            key, value = next(iter(entry.items()))
            if key == "image":
                pil_img = ComfyUIClient.read_image(value)
                return _pil_to_tensor(pil_img)
            if key == "text":
                return str(value)
            return value
        return entry

    def get_output(self, results: dict, node: str = ""):
        if not isinstance(results, dict):
            raise RuntimeError("results 格式不正确")

        outputs = results.get("outputs") or {}
        node_id = self._resolve_node_id(results, node, outputs)

        node_outputs = outputs.get(node_id)
        if node_outputs is None:
            node_outputs = results.get("all_outputs", {}).get(node_id)
        if node_outputs is None:
            raise RuntimeError(f"未找到节点 {node_id} 的输出")

        converted = [self._convert_entry(entry) for entry in node_outputs]
        return (converted,)
