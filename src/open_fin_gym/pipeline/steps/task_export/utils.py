import re


def slugify(name: str) -> str:
    """
    Convert a task name into a directory name

    Args:
        name: Task name, which the LLM is free to choose

    Returns:
        Name reduced to lowercase words joined by underscores
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "task"
