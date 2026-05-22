import json
from typing import Any, Dict


class JsonHelper:
    """简易 JSON 读写助手。"""

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(path: str, data: Any) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
