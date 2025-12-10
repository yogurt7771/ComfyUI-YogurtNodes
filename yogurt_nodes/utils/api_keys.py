from pathlib import Path
import json


def load_api_keys():
    api_key_path = Path(__file__).parent.parent / "llm" / "api_key.json"
    if api_key_path.exists():
        with open(api_key_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
