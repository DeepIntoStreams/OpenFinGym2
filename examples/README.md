# RL and SFT Training Examples

These scripts demonstrate the use of OpenFinGym generated tasks for SFT and RL training.

They are largely adapted from
[SkyRLs Harbor integration examples](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train_integrations/harbor).

Harbor supports several integrations with their task format see the
[Harbor docs](https://www.harborframework.com/docs) and
[Harbor Cookbook Repo](https://github.com/harbor-framework/harbor-cookbook/tree/main)
for more details.

## Running

### RL

The reinforcement learning example can be run using

```commandline
uv run examples/rl/main.py
```

The training process is configured in `examples/rl/default_config.yaml`.

### SFT

Adapted from [Harbor docs](https://www.harborframework.com/docs/training-workflows/sft)

Tasks can be run using harbor, and traces exported for sft

```commandline
harbor run \
    -p "<path/to/dataset>" \
    -m "<model>" \
    -a "<agent>" \
    --export-traces \
    --export-sharegpt \
    --export-episodes last \
```
