import os
import json
from contextlib import contextmanager
from typing import Optional


env_proxies = [
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
]


def get_proxy_url(proxy_url: str = "") -> Optional[str]:
    """
    获取代理URL，支持三种方式获取代理设置

    优先级：
    1. 直接传入的proxy_url参数
    2. api_key.json文件中的proxy配置
    3. 环境变量ALL_PROXY, HTTP_PROXY, HTTPS_PROXY

    Args:
        proxy_url (str): 直接传入的代理URL

    Returns:
        str: 代理URL，如果没有配置则返回None
    """
    # 1. 优先使用直接传入的参数
    if proxy_url and proxy_url.strip():
        return proxy_url.strip()

    # 2. 尝试从api_key.json文件读取
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_key_path = os.path.join(current_dir, "api_key.json")
        if os.path.exists(api_key_path):
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_keys = json.load(f)
                if "proxy" in api_keys and api_keys["proxy"]:
                    return api_keys["proxy"].strip()
    except Exception:
        pass

    # 3. 尝试从环境变量读取
    for env_var in env_proxies:
        env_proxy = os.getenv(env_var, "").strip()
        if env_proxy:
            return env_proxy

    return None


@contextmanager
def proxy_env(proxy_url: str = ""):
    """
    临时设置代理环境变量的上下文管理器

    Args:
        proxy_url (str): 代理URL，格式: protocol://user:pass@addr:port
                        支持 http, https, socks5, socks5h

    使用方法:
        with proxy_env("http://127.0.0.1:8080"):
            # 在这个作用域内，ALL_PROXY环境变量被临时设置
            response = requests.get("https://api.example.com")
    """
    actual_proxy_url = get_proxy_url(proxy_url)

    # 保存原始环境变量
    original_proxies = {key: os.environ.get(key) for key in env_proxies}

    try:
        # 如果有代理URL，设置环境变量
        if actual_proxy_url:
            print(f"Setting proxy: {actual_proxy_url}")
            for key in env_proxies:
                os.environ[key] = actual_proxy_url

        yield actual_proxy_url

    finally:
        # 恢复原始环境变量
        for key in env_proxies:
            original = original_proxies.get(key)
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


class SetProxyEnv:
    """
    代理环境设置上下文管理器。

    逻辑变更：
    - 如果传入有效的 proxy_url：备份旧环境 -> 设置新代理 -> 退出时还原。
    - 如果传入 None 或 空字符串：直接跳过（Pass-through），不修改任何环境变量，完全使用系统原有状态。
    """

    def __init__(self, proxy_url: str = None):
        self.proxy_url = proxy_url
        # Linux下必须同时覆盖全大写和全小写
        self.target_keys = env_proxies
        self.original_values = {}
        # 标志位：记录是否真的修改了环境
        self.applied = False

    def __enter__(self):
        # 1. 检查逻辑：如果是 None 或 空字符串，直接跳过
        if not self.proxy_url:
            return self

        # 2. 标记为“已应用修改”
        self.applied = True

        # 3. 备份当前状态
        for key in self.target_keys:
            self.original_values[key] = os.environ.get(key)

        # 4. 设置新代理
        for key in self.target_keys:
            os.environ[key] = self.proxy_url

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 5. 如果当初没修改（applied 为 False），退出时也什么都不做
        if not self.applied:
            return

        # 6. 恢复现场
        for key in self.target_keys:
            original = self.original_values.get(key)

            if original is None:
                # 之前不存在，现在删掉
                os.environ.pop(key, None)
            else:
                # 之前存在，恢复原值
                os.environ[key] = original
