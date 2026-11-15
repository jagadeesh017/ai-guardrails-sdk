from pathlib import Path
from typing import Any

import yaml

from ..core.policy import Policy


class PolicyLoader:
    @staticmethod
    def load(path: str | Path) -> Policy:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
        elif path.suffix == ".json":
            import json
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        return PolicyLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Policy:
        return Policy.model_validate(data)

    @staticmethod
    def from_yaml_string(content: str) -> Policy:
        data = yaml.safe_load(content)
        return PolicyLoader.from_dict(data)