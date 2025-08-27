import os
import json
from contextlib import contextmanager
from typing import Optional


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
    env_proxies = ['ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy']
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
    original_all_proxy = os.environ.get('ALL_PROXY')
    original_http_proxy = os.environ.get('HTTP_PROXY') 
    original_https_proxy = os.environ.get('HTTPS_PROXY')
    
    try:
        # 如果有代理URL，设置环境变量
        if actual_proxy_url:
            print(f"Setting proxy: {actual_proxy_url}")
            os.environ['ALL_PROXY'] = actual_proxy_url
            os.environ['HTTP_PROXY'] = actual_proxy_url
            os.environ['HTTPS_PROXY'] = actual_proxy_url
        
        yield actual_proxy_url
        
    finally:
        # 恢复原始环境变量
        if original_all_proxy is not None:
            os.environ['ALL_PROXY'] = original_all_proxy
        else:
            os.environ.pop('ALL_PROXY', None)
            
        if original_http_proxy is not None:
            os.environ['HTTP_PROXY'] = original_http_proxy
        else:
            os.environ.pop('HTTP_PROXY', None)
            
        if original_https_proxy is not None:
            os.environ['HTTPS_PROXY'] = original_https_proxy
        else:
            os.environ.pop('HTTPS_PROXY', None)