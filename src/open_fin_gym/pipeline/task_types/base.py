from dataclasses import dataclass


@dataclass
class TaskTypeParams:
    # Task-type name/identifier
    name: str
    # Example of the task provided to task generation agent
    example: str
    # Output files required by this task-type
    required_groups: set[str]
    # Whether row i of the user output corresponds to row i of the test target,
    # which decides whether the verifier can compare them row by row
    row_correspondence: str
