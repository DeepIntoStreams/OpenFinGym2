from open_fin_gym.pipeline.steps.task_extraction.utils import TaskSpecification


def structural_issues(spec: TaskSpecification) -> list[str]:
    """Non-LLM sanity checks on an extracted task spec."""
    issues = []
    groups = {
        "training_inputs": spec.training_inputs,
        "training_targets": spec.training_targets,
        "test_inputs": spec.test_inputs,
        "test_targets": spec.test_targets,
        "test_outputs": spec.test_outputs,
    }

    for name, datasets in groups.items():
        if not datasets:
            issues.append(f"{name} is empty")

    if not spec.metrics:
        issues.append("assessment_metrics is empty")

    filenames = [d.filename for datasets in groups.values() for d in datasets]
    duplicates = {x for x in filenames if filenames.count(x) > 1}
    if duplicates:
        issues.append(f"duplicate dataset filenames across roles: {sorted(duplicates)}")

    known_filenames = {d.filename for d in spec.test_outputs + spec.test_targets}
    for metric in spec.metrics:
        missing = [x for x in metric.input_datasets if x not in known_filenames]
        if missing:
            issues.append(
                f"metric '{metric.name}' references unknown dataset(s): {missing}"
            )

    return issues
