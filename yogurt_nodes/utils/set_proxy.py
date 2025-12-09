import os


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
        self.target_keys = ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]
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
