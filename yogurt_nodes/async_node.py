"""把同步节点批量包装成异步执行版本的工厂。

ComfyUI 的执行引擎原生支持把 V1 节点的 FUNCTION 定义为 ``async def``
（execution.py 中 ``inspect.iscoroutinefunction(f)`` 命中后会用 asyncio.Task
并发调度）；本仓库的 V3 适配器（v3_node_adapter / v3_llm_adapter）的 ``execute``
也是 ``async``，且会对返回值 ``await``（``inspect.isawaitable``）。因此无论节点
最终是否被包成 V3，只要其 FUNCTION 是协程函数即可异步执行，无需迁移到 v3。

``make_async_node`` 子类化一个已有同步节点，新增一个协程 FUNCTION：把原同步
方法整体丢到工作线程（``asyncio.to_thread``）执行，等待网络/IO/推理期间不占用
事件循环。原同步方法内部若使用可中断 HTTP 客户端，其取消能力保持不变。
"""

from __future__ import annotations

import asyncio
from typing import Optional, Type


def make_async_node(
    base_cls: Type,
    *,
    name: Optional[str] = None,
    node_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Type:
    """基于同步节点 ``base_cls`` 生成功能一致的异步执行子类。

    - 读取 ``base_cls.FUNCTION``（可继承）作为被包装的同步方法名；
    - 新增协程方法 ``<orig>_async`` 并将其设为新的 ``FUNCTION``；
    - ``_NODE_NAME`` 默认在原名后追加 " Async"。
    """
    orig_func = base_cls.FUNCTION
    async_func = f"{orig_func}_async"
    cls_name = name or f"{base_cls.__name__}Async"

    async def _async_method(self, **kwargs):
        # self.<orig_func> 解析到父类的同步实现（子类未覆盖它，只新增了协程方法），
        # 整体放到线程执行以释放事件循环。
        return await asyncio.to_thread(getattr(self, orig_func), **kwargs)

    _async_method.__name__ = async_func
    _async_method.__qualname__ = f"{cls_name}.{async_func}"

    base_node_name = getattr(base_cls, "_NODE_NAME", base_cls.__name__)
    base_desc = getattr(base_cls, "DESCRIPTION", "") or base_node_name

    namespace = {
        async_func: _async_method,
        "FUNCTION": async_func,
        "_NODE_NAME": node_name or f"{base_node_name} Async",
        "DESCRIPTION": description or f"{base_desc}（异步执行，等待返回时不阻塞 ComfyUI 事件循环）",
        "__doc__": f"{base_cls.__name__} 的异步执行版本（功能一致）。",
        "__module__": base_cls.__module__,
    }
    return type(cls_name, (base_cls,), namespace)
