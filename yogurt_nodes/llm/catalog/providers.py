"""供应商声明。route_key 决定模型声明里用哪一组路由；兼容渠道与官方共用路由。"""

from __future__ import annotations

from ..framework import ProviderSpec, boolean, integer, register_provider, text
from .common import api_key_field, base_url_field, proxy_field, timeout_field

GOOGLE_AI = register_provider(
    ProviderSpec(
        kind="google_ai",
        display_name="Google AI Studio",
        route_key="gemini",
        fields=(api_key_field("gemini", "GEMINI_API_KEY"), proxy_field(), timeout_field()),
        custom_protocols={"text": "gemini", "image": "gemini"},
    )
)

GENAI_COMPATIBLE = register_provider(
    ProviderSpec(
        kind="genai_compatible",
        display_name="Google GenAI Compatible",
        route_key="gemini",
        fields=(
            base_url_field(tooltip="Base URL of a Gemini-API-compatible endpoint."),
            api_key_field("gemini", "GEMINI_API_KEY"),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"text": "gemini", "image": "gemini"},
    )
)

VERTEX_AI = register_provider(
    ProviderSpec(
        kind="vertex_ai",
        display_name="Vertex AI",
        route_key="gemini",
        fields=(
            text("api_key", tooltip="Vertex AI express-mode API key. Leave empty to use service-account credentials."),
            text(
                "credentials_json",
                multiline=True,
                tooltip="Service-account JSON. Leave empty to use 'vertex_ai_json' from api_key.json "
                "or GOOGLE_APPLICATION_CREDENTIALS.",
            ),
            text("project_id", tooltip="Leave empty to use 'vertex_ai_project' from api_key.json."),
            text("location", tooltip="Region such as global or us-central1; empty uses 'vertex_ai_region'."),
            base_url_field(),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"text": "gemini", "image": "gemini"},
    )
)

OPENAI = register_provider(
    ProviderSpec(
        kind="openai",
        display_name="OpenAI",
        route_key="openai",
        fields=(api_key_field("openai", "OPENAI_API_KEY"), proxy_field(), timeout_field()),
        custom_protocols={"text": "openai_chat", "image": "openai_images"},
        connection={
            "base_url": "https://api.openai.com/v1",
            "base_url_json": ("openai_base_url",),
            "base_url_env": ("OPENAI_BASE_URL",),
            "key_json": ("openai",),
            "key_env": ("OPENAI_API_KEY",),
            "max_tokens_field": "max_completion_tokens",
        },
    )
)

OPENAI_COMPATIBLE = register_provider(
    ProviderSpec(
        kind="openai_compatible",
        display_name="OpenAI Compatible",
        route_key="openai",
        fields=(
            base_url_field(tooltip="Base URL including the version path, e.g. https://example.com/v1."),
            text("api_key", tooltip="API key for this endpoint (required; official keys are never sent here)."),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"text": "openai_chat", "image": "openai_images"},
    )
)

OPENROUTER = register_provider(
    ProviderSpec(
        kind="openrouter",
        display_name="OpenRouter",
        route_key="openrouter",
        fields=(
            api_key_field("openrouter", "OPENROUTER_API_KEY"),
            text("provider_order", tooltip="Comma-separated upstream providers to try in order, e.g. google-vertex,openai."),
            boolean("allow_fallbacks", True, tooltip="Allow other upstream providers when provider_order fails."),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"text": "openai_chat", "image": "openrouter_images"},
        connection={
            "base_url": "https://openrouter.ai/api/v1",
            "key_json": ("openrouter",),
            "key_env": ("OPENROUTER_API_KEY",),
        },
    )
)

XAI = register_provider(
    ProviderSpec(
        kind="xai",
        display_name="xAI",
        route_key="xai",
        fields=(api_key_field("xai", "XAI_API_KEY"), proxy_field(), timeout_field()),
        custom_protocols={"text": "openai_chat", "image": "xai_images"},
        connection={
            "base_url": "https://api.x.ai/v1",
            "base_url_json": ("xai_base_url", "grok_base_url"),
            "base_url_env": ("XAI_BASE_URL", "GROK_BASE_URL"),
            "key_json": ("xai", "grok"),
            "key_env": ("XAI_API_KEY", "GROK_API_KEY"),
        },
    )
)

GRSAI = register_provider(
    ProviderSpec(
        kind="grsai",
        display_name="GRSAI",
        route_key="grsai",
        fields=(
            api_key_field("grsai", "GRSAI_API_KEY"),
            base_url_field(),
            integer("max_wait_seconds", 600, min=30, max=3600, tooltip="Maximum time to wait for the draw task."),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"image": "grsai_draw"},
        connection={
            "base_url": "https://grsaiapi.com",
            "base_url_json": ("grsai_base_url",),
            "base_url_env": ("GRSAI_BASE_URL",),
            "key_json": ("grsai", "grsai_api_key"),
            "key_env": ("GRSAI_API_KEY",),
        },
    )
)

FREEDOMGPT = register_provider(
    ProviderSpec(
        kind="freedomgpt",
        display_name="FreedomGPT",
        route_key="freedomgpt",
        fields=(api_key_field("freedomgpt", "FREEDOMGPT_API_KEY"), proxy_field(), timeout_field()),
        custom_protocols={"text": "freedomgpt_chat", "image": "freedomgpt_image"},
        connection={
            "base_url": "https://chat.freedomgpt.com/api/v1",
            "key_json": ("freedomgpt",),
            "key_env": ("FREEDOMGPT_API_KEY",),
        },
    )
)

VOLCENGINE_ARK = register_provider(
    ProviderSpec(
        kind="volcengine_ark",
        display_name="Volcengine Ark",
        route_key="volcengine_ark",
        fields=(
            api_key_field("seedream", "ARK_API_KEY"),
            base_url_field(tooltip="Default https://ark.cn-beijing.volces.com/api/v3."),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"text": "openai_chat", "image": "ark_images"},
        connection={
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "key_json": ("seedream", "ark"),
            "key_env": ("ARK_API_KEY", "SEEDREAM_API_KEY"),
        },
    )
)

BYTEPLUS_ARK = register_provider(
    ProviderSpec(
        kind="byteplus_ark",
        display_name="BytePlus ModelArk",
        route_key="byteplus_ark",
        fields=(
            api_key_field("byteplus", "BYTEPLUS_API_KEY"),
            base_url_field(tooltip="Default https://ark.ap-southeast.bytepluses.com/api/v3."),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"text": "openai_chat", "image": "ark_images"},
        connection={
            "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
            "key_json": ("byteplus",),
            "key_env": ("BYTEPLUS_API_KEY",),
        },
    )
)

DASHSCOPE = register_provider(
    ProviderSpec(
        kind="dashscope",
        display_name="Alibaba DashScope",
        route_key="dashscope",
        fields=(
            api_key_field("dashscope", "DASHSCOPE_API_KEY"),
            base_url_field(tooltip="Default https://dashscope.aliyuncs.com/api/v1; "
                           "international: https://dashscope-intl.aliyuncs.com/api/v1."),
            proxy_field(),
            timeout_field(),
        ),
        custom_protocols={"image": "dashscope_image"},
        connection={
            "base_url": "https://dashscope.aliyuncs.com/api/v1",
            "base_url_json": ("dashscope_base_url", "qwen_base_url", "wan_base_url"),
            "base_url_env": ("DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "WAN_BASE_URL"),
            "key_json": ("dashscope", "qwen", "wan"),
            "key_env": ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "WAN_API_KEY"),
        },
    )
)


def _task_timeout():
    return integer("task_timeout", 900, min=30, max=7200, tooltip="Maximum time to wait for the upscale task.")


TOPAZ = register_provider(
    ProviderSpec(
        kind="topaz",
        display_name="Topaz Labs",
        route_key="topaz",
        fields=(api_key_field("topaz", "TOPAZ_API_KEY"), base_url_field(), _task_timeout(), proxy_field(),
                timeout_field()),
        custom_protocols={"upscale": "topaz"},
        connection={
            "base_url": "https://api.topazlabs.com/image/v1",
            "base_url_json": ("topaz_base_url",),
            "base_url_env": ("TOPAZ_BASE_URL",),
            "key_json": ("topaz",),
            "key_env": ("TOPAZ_API_KEY",),
        },
    )
)

MAGNIFIC = register_provider(
    ProviderSpec(
        kind="magnific",
        display_name="Magnific",
        route_key="magnific",
        fields=(api_key_field("magnific", "MAGNIFIC_API_KEY"), base_url_field(), _task_timeout(), proxy_field(),
                timeout_field()),
        connection={
            "base_url": "https://api.magnific.com",
            "base_url_json": ("magnific_base_url",),
            "base_url_env": ("MAGNIFIC_BASE_URL",),
            "key_json": ("magnific",),
            "key_env": ("MAGNIFIC_API_KEY",),
        },
    )
)
