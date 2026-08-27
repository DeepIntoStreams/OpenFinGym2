# RL and SFT Training Examples

These scripts demonstrate the use of OpenFinGym generated tasks for SFT and RL training.

They are largely adapted from
[SkyRLs Harbor integration examples](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train_integrations/harbor).

Harbor supports several integrations with their task format see the
[Harbor docs](https://www.harborframework.com/docs) for more details.

## Running

### RL

The reinforcement learning example can be run using

```commandline
uv run examples/rl/main.py
```

The training process is configured in `examples/rl/default_config.yaml`.
