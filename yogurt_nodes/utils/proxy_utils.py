import os
import threading


env_proxies = [
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
]

_proxy_env_lock = threading.RLock()


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
        self._lock_acquired = False

    def __enter__(self):
        # 1. 检查逻辑：如果是 None 或 空字符串，直接跳过
        if not self.proxy_url:
            return self

        _proxy_env_lock.acquire()
        self._lock_acquired = True
        print(f"[YogurtNodes] Setting proxy environment to: {self.proxy_url}")

        try:
            # 2. 备份当前状态
            for key in self.target_keys:
                self.original_values[key] = os.environ.get(key)

            # 3. 设置新代理
            for key in self.target_keys:
                os.environ[key] = self.proxy_url

            # 4. 标记为“已应用修改”
            self.applied = True
        except Exception:
            for key, original in self.original_values.items():
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original
            if self._lock_acquired:
                self._lock_acquired = False
                _proxy_env_lock.release()
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 5. 如果当初没修改（applied 为 False），退出时也什么都不做
        if not self.applied:
            if self._lock_acquired:
                self._lock_acquired = False
                _proxy_env_lock.release()
            return

        print(f"[YogurtNodes] Restoring original proxy environment.")

        try:
            # 6. 恢复现场
            for key in self.target_keys:
                original = self.original_values.get(key)

                if original is None:
                    # 之前不存在，现在删掉
                    os.environ.pop(key, None)
                else:
                    # 之前存在，恢复原值
                    os.environ[key] = original
        finally:
            self.applied = False
            if self._lock_acquired:
                self._lock_acquired = False
                _proxy_env_lock.release()
