from dataclasses import dataclass
from typing import Any


@dataclass
class TaskRoutingConfig:
    llm: Any
    bundles_path: str
    export_path: str
