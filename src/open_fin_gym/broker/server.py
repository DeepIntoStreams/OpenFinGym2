import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from open_fin_gym.realtime.config import TradingConfig
from open_fin_gym.realtime.data_providers.alpaca import AlpacaProvider
from open_fin_gym.realtime.data_providers.binance import BinanceProvider
from open_fin_gym.realtime.tasks.realtime_trading_task import (
    RealtimeTradingTask,
)

CONFIG_PATH = Path(os.environ.get("BROKER_CONFIG", "/broker/episode.json"))
LEDGER_PATH = Path(os.environ.get("BROKER_LEDGER", "/ledger/episode.jsonl"))

PROVIDERS = {"alpaca": AlpacaProvider, "binance": BinanceProvider}


def build_task(spec: dict[str, Any]) -> Any:
    """
    Build the task an episode configuration describes

    Args:
        spec: Parsed episode configuration

    Returns:
        The task instance the broker drives
    """
    kind = spec["kind"]
    provider = PROVIDERS[spec["provider"]]()
    if kind == "realtime_trading":
        return RealtimeTradingTask(
            provider=provider,
            symbols=spec["symbols"],
            trading_config=TradingConfig(**spec.get("trading", {})),
            context_resolutions=spec.get("context_resolutions"),
            data_resolution=spec.get("data_resolution"),
            max_steps=int(spec.get("max_steps", 0)),
            target_symbols=spec.get("target_symbols"),
            initial_capital=float(spec.get("initial_capital", 100000.0)),
        )
    raise ValueError(f"Unsupported task kind {kind}")


def create_app() -> FastAPI:
    spec = json.loads(CONFIG_PATH.read_text())
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    task = build_task(spec)
    actions: list[Any] = []
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/reset")
    def reset() -> dict:
        actions.clear()
        LEDGER_PATH.write_text("")
        return task.reset()

    @app.post("/step")
    def step(action: dict) -> dict:
        try:
            observation, reward, done, info = task.step(action)
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
        actions.append(action)
        with LEDGER_PATH.open("a") as f:
            f.write(
                json.dumps(
                    {"action": action, "reward": reward, "info": info}, default=str
                )
                + "\n"
            )
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }

    @app.get("/score")
    def score() -> dict:
        return task.evaluate(actions)

    return app


app = create_app()
