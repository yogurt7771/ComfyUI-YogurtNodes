import base64
import io
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys


def image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为base64编码"""
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format="JPEG", quality=95)
    img_bytes = img_bytes_io.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


class SeeDreamClient:
    """
    豆包-SeeDream客户端，用于调用火山引擎豆包图像生成API
    """

    def __init__(self, api_key: str = "", proxy_url: str = "", timeout: int = 0):
        self.api_key = self._get_api_key(api_key)
        self.proxy_url = proxy_url
        self.timeout = timeout

    def _get_api_key(self, api_key: str) -> str:
        """获取API密钥，按优先级顺序：参数 > 配置文件 > 环境变量"""
        if api_key:
            return api_key

        # 尝试从配置文件读取
        try:
            api_keys = load_api_keys()
            if "seedream" in api_keys:
                return api_keys["seedream"]
        except Exception:
            pass

        # 尝试从环境变量读取
        env_key = os.getenv("ARK_API_KEY") or os.getenv("SEEDREAM_API_KEY")
        if env_key:
            return env_key

        return ""

    def _prepare_headers(self) -> Dict[str, str]:
        """准备请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _prepare_proxies(self) -> Optional[Dict[str, str]]:
        """准备代理设置"""
        if self.proxy_url:
            return {"http": self.proxy_url, "https": self.proxy_url}
        return None

    def _upload_image_to_base64_url(self, image: Image.Image) -> str:
        """将PIL图像转换为base64 data URL"""
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format="JPEG", quality=95)
        img_bytes = img_bytes_io.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    def _download_image(self, url: str) -> Image.Image:
        """从URL下载图像"""
        response = requests.get(url, proxies=self._prepare_proxies(), timeout=self.timeout if self.timeout > 0 else None)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    def generate_image(
        self,
        model: str,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        size: str = "2K",
        sequential_image_generation: str = "disabled",
        max_images: int = 1,
        stream: bool = False,
        response_format: str = "url",
        watermark: bool = True,
        region: str = "cn-beijing",
        retry_count: int = 3,
        seed: int = -1,
        extra: dict | None = None,
    ) -> Tuple[List[Image.Image], str]:
        """
        生成图像

        Args:
            model: 模型名称
            prompt: 提示词
            images: 输入图像列表（用于图生图）
            size: 图像尺寸
            sequential_image_generation: 序列图像生成模式
            max_images: 最大图像数量
            stream: 是否流式返回
            response_format: 响应格式
            watermark: 是否添加水印
            retry_count: 重试次数
            seed: 随机种子
        Returns:
            (生成的图像列表, 响应文本)
        """

        base_url = f"https://ark.{region}.volces.com/api/v3/images/generations"
        # 准备请求数据
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "sequential_image_generation": sequential_image_generation,
            "stream": stream,
            "response_format": response_format,
            "watermark": watermark,
            "seed": seed,
        }

        # 处理输入图像
        if images:
            if len(images) == 1:
                data["image"] = self._upload_image_to_base64_url(images[0])
            else:
                data["image"] = [
                    self._upload_image_to_base64_url(img) for img in images
                ]

        # 处理序列图像生成选项
        if sequential_image_generation == "auto" and max_images > 1:
            data["sequential_image_generation_options"] = {"max_images": max_images}

        # 合并额外参数
        if extra:
            data.update(extra)

        # 执行请求，支持重试
        last_error = None
        for attempt in range(retry_count):
            try:
                print(f"[SeeDream] 发送请求 (尝试 {attempt + 1}/{retry_count})")
                print(f"[SeeDream] 提示词: {prompt[:100]}...")

                model_management.throw_exception_if_processing_interrupted()

                # 使用requests进行API调用
                headers = self._prepare_headers()
                proxies = self._prepare_proxies()

                response = requests.post(
                    base_url,
                    headers=headers,
                    json=data,
                    proxies=proxies,
                    timeout=self.timeout if self.timeout > 0 else None,
                )

                response.raise_for_status()
                result = response.json()

                print(f"[SeeDream] 请求成功")

                # 解析响应
                generated_images = []
                response_text = json.dumps(result, ensure_ascii=False, indent=2)

                if "data" in result:
                    for item in result["data"]:
                        if response_format == "url" and "url" in item:
                            # 下载图像
                            img = self._download_image(item["url"])
                            generated_images.append(img)
                        elif response_format == "b64_json" and "b64_json" in item:
                            # 解码base64图像
                            img_data = base64.b64decode(item["b64_json"])
                            img = Image.open(io.BytesIO(img_data))
                            generated_images.append(img)

                if not generated_images:
                    print(f"[SeeDream] 警告: 未找到生成的图像")
                    # 创建一个空白图像作为占位符
                    placeholder = Image.new("RGB", (512, 512), color="white")
                    generated_images.append(placeholder)

                return generated_images, response_text

            except Exception as e:
                last_error = e
                print(
                    f"[SeeDream] 请求失败 (尝试 {attempt + 1}/{retry_count}): {str(e)}"
                )
                if attempt < retry_count - 1:
                    time.sleep(2**attempt)  # 指数退避

        # 所有重试都失败了
        error_msg = f"SeeDream API 请求失败，已重试 {retry_count} 次。最后错误: {str(last_error)}"
        print(f"[SeeDream] {error_msg}")

        # 返回空白图像和错误信息
        placeholder = Image.new("RGB", (512, 512), color="red")
        return [placeholder], error_msg

    @classmethod
    def get_sequential_modes(cls) -> List[str]:
        """获取支持的序列生成模式"""
        return ["disabled", "auto"]
