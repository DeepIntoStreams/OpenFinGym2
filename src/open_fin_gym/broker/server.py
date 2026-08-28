import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from open_fin_gym.realtime.tasks.offline_stock_forecasting import (
    OfflineStockForecasting,
)
from open_fin_gym.realtime.tasks.offline_stock_trading import (
    OfflineStockTrading,
)
from open_fin_gym.realtime.tasks.realtime_stock_forecasting import (
    RealtimeStockForecasting,
)
from open_fin_gym.realtime.tasks.realtime_stock_trading import (
    RealtimeStockTrading,
)

CONFIG_PATH = Path(os.environ.get("BROKER_CONFIG", "/broker/episode.json"))
LEDGER_PATH = Path(os.environ.get("BROKER_LEDGER", "/ledger/episode.jsonl"))

TASKS = {
    "realtime_trading": RealtimeStockTrading,
    "realtime_forecasting": RealtimeStockForecasting,
    "offline_trading": OfflineStockTrading,
    "offline_forecasting": OfflineStockForecasting,
}


def create_app() -> FastAPI:
    spec = json.loads(CONFIG_PATH.read_text())
    if spec["kind"] not in TASKS:
        raise ValueError(f"Unsupported task kind {spec['kind']}")
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    task = TASKS[spec["kind"]](config=spec.get("config", {}))
    actions: list[Any] = []
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/reset")
    def reset() -> Any:
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
        record = {"action": action, "reward": reward, "info": info}
        with LEDGER_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
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
