from dataclasses import dataclass


@dataclass
class TaskConfig:
    org_name: str


@dataclass
class TaskExportConfig:
    export_path: str
    templates_path: str
    task_config: TaskConfig
