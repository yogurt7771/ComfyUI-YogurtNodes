# LLM 节点扩展指南

新节点由三层组成，彼此只通过名字关联：

| 层 | 位置 | 作用 |
| --- | --- | --- |
| 协议 | `protocols/*.py` | 一种 API 格式的实现（如 `gemini`、`openai_chat`、`openai_images`、`openrouter_images`、`ark_images`、`dashscope_image`、`topaz`），用 `@register_protocol("名字")` 注册 |
| 供应商 | `catalog/providers.py` | 一个接入渠道：节点上显示的字段、默认地址、`api_key.json` 键名与环境变量、使用哪组模型路由 |
| 模型 | `catalog/<厂商>.py` | 模型系列（`FamilySpec`）及其版本（`ModelSpec`）：参数声明 + 在各供应商下的路由 |

节点类（`provider_nodes.py`、`image_nodes.py`、`text_nodes.py`、`upscale_nodes.py`）只是薄壳，指向上面的声明。公共逻辑在 `framework/`：参数到 V3 输入的转换、DynamicCombo 版本切换、Autogrow 图片输入、默认供应商、路由解析、重试与报错。

## 给已有系列加一个模型版本

在对应系列的 `models` 里加一条 `ModelSpec`，不需要改节点类。以 Nano Banana 为例（`catalog/gemini.py`）：

```python
ModelSpec(
    "Nano Banana 3",                                   # 下拉里显示的名字
    routes=_gemini("gemini-nano-banana-3",             # Google 官方模型 ID
                   openrouter="google/gemini-nano-banana-3"),
    params=(
        aspect_ratio(ASPECT_RATIOS_EXTENDED),
        _resolution(["1K", "2K", "4K"]),
        _thinking_level(["MINIMAL", "HIGH"], "MINIMAL"),
        _modalities(),
    ),
),
```

- `routes` 的键是供应商的 `kind`（如 `grsai`）或 `route_key`（如 `gemini`、`openai`、`openrouter`）。查找顺序是先 `kind` 后 `route_key`，所以兼容渠道会自动复用官方路由，个别渠道模型 ID 不同时再按 `kind` 单独写。
- `params` 只写这个版本真正支持的参数；切换版本时节点上的输入会随之变化。
- `max_images` 可限制该版本的参考图数量。

## 新增模型系列

1. 在 `catalog/` 中写一个 `FamilySpec`（`task` 为 `image` / `text` / `upscale`，并给出 `default_provider`）。
2. 在对应的 `*_nodes.py` 中加一个薄节点类：

   ```python
   class FluxGenerateImage(ModelFamilyNode):
       _NODE_NAME = "FLUX"
       DESCRIPTION = "Black Forest Labs FLUX image generation."
       FAMILY = bfl.FLUX
   ```

3. 在 `llm/__init__.py` 中导出该类。节点 ID 为 `Yogurt` + 类名，确定后不要再改，否则已保存的工作流会找不到节点。
4. 运行 `python tools/generate_readme.py` 更新 README；类的 docstring 按 `tools/sync_node_docstrings.py` 的规则生成。

`custom_params` 不为空时，下拉里会多出 `Custom model` 选项，用户可以直接填写模型 ID。

## 新增供应商

### 接口格式已有协议支持（例如又一个 OpenAI 兼容中转）

1. 在 `catalog/providers.py` 中注册：

   ```python
   EXAMPLE = register_provider(
       ProviderSpec(
           kind="example",
           display_name="Example",
           route_key="openai",                  # 复用 OpenAI 系模型的路由
           fields=(api_key_field("example", "EXAMPLE_API_KEY"), proxy_field(), timeout_field()),
           custom_protocols={"text": "openai_chat", "image": "openai_images"},
           connection={
               "base_url": "https://api.example.com/v1",
               "key_json": ("example",),
               "key_env": ("EXAMPLE_API_KEY",),
           },
       )
   )
   ```

   `connection` 里不写 `key_json` / `key_env` 时，key 只能在节点中填写。第三方中转应保持这样，避免把官方 key 发给第三方。

2. 在 `provider_nodes.py` 中加一个 `ProviderNode` 子类，并在 `llm/__init__.py` 中导出。
3. 如果这个渠道上的模型 ID 和官方不同，给相关 `ModelSpec` 的 `routes` 加上以该 `kind` 为键的路由。

### 全新的接口格式

在 `protocols/` 下新建模块并在 `protocols/__init__.py` 中导入：

```python
@register_protocol("example_images")
async def run_example(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)                         # 解析 key、base_url，创建可中断的 HTTP 客户端
    payload = {"model": route.model, "prompt": image_prompt(request)}
    ...
    body = await request_json(conn, conn.url("images"), payload, f"{conn.name} image")
    return GenerationResult(images=[decode_image(item["b64_json"]) for item in body["data"]])
```

协议只负责把统一参数翻译成该 API 的请求并解析结果；重试、超时提示、报错格式由框架统一处理。HTTP 请求请使用 `protocols/http.py` 中的工具，这样 ComfyUI 的取消操作能中断请求。

## 统一参数名

模型声明使用下列参数名，各协议负责翻译成对应 API 的字段：

| 参数 | 含义 |
| --- | --- |
| `aspect_ratio`、`resolution` | 宽高比、分辨率档位（`auto` 表示不发送） |
| `size`、`custom_width`、`custom_height` | 具体尺寸（`1024x1024`、`1024*1024` 或带说明的预设名），`Custom` 时使用宽高 |
| `quality`、`background`、`output_format`、`output_compression`、`moderation`、`n` | 生图通用选项 |
| `thinking_level`、`thinking_budget`、`reasoning_effort`、`verbosity` | 推理控制；`default` 表示不发送 |
| `temperature`、`top_p`、`max_output_tokens` | 采样与输出长度 |
| `negative_prompt`、`seed`、`extra` | 反向提示词、种子、透传 JSON |

## 验证

- 加载检查：在 ComfyUI 环境中导入插件，确认新节点出现在 `NODE_CLASS_MAPPINGS` 中，且 `GET_NODE_INFO_V1()` 正常。
- 离线请求检查：拦截 HTTP 层，调用节点的 `execute`，核对请求地址和请求体，不产生真实调用。
