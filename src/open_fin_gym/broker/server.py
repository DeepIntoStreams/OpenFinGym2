import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from open_fin_gym.realtime.tasks.offline_crypto_forecasting import (
    OfflineCryptoForecasting,
)
from open_fin_gym.realtime.tasks.offline_crypto_trading import (
    OfflineCryptoTrading,
)
from open_fin_gym.realtime.tasks.offline_stock_forecasting import (
    OfflineStockForecasting,
)
from open_fin_gym.realtime.tasks.offline_stock_trading import (
    OfflineStockTrading,
)
from open_fin_gym.realtime.tasks.realtime_crypto_forecasting import (
    RealtimeCryptoForecasting,
)
from open_fin_gym.realtime.tasks.realtime_crypto_trading import (
    RealtimeCryptoTrading,
)
from open_fin_gym.realtime.tasks.realtime_polymarket import RealtimePolymarket
from open_fin_gym.realtime.tasks.realtime_stock_forecasting import (
    RealtimeStockForecasting,
)
from open_fin_gym.realtime.tasks.realtime_stock_trading import (
    RealtimeStockTrading,
)
from open_fin_gym.realtime.tasks.realtime_stock_trading_alpaca_paper import (
    RealtimeStockTradingAlpacaPaper,
)


def as_json(value: Any) -> Any:
    """
    Convert arrays to plain lists so the payload can be serialised

    Args:
        value: Feature or target data from a task

    Returns:
        The same data using only JSON types
    """
    if isinstance(value, dict):
        return {k: as_json(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def as_arrays(value: Any) -> Any:
    """
    Convert lists back to arrays for the task's own scoring code

    Args:
        value: Predictions as submitted by the agent

    Returns:
        The same data with lists replaced by arrays
    """
    if isinstance(value, dict):
        return {k: as_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


CONFIG_PATH = Path(os.environ.get("BROKER_CONFIG", "/broker/episode.json"))
LEDGER_PATH = Path(os.environ.get("BROKER_LEDGER", "/ledger/episode.jsonl"))

TASKS = {
    "realtime_trading": RealtimeStockTrading,
    "realtime_forecasting": RealtimeStockForecasting,
    "offline_trading": OfflineStockTrading,
    "offline_forecasting": OfflineStockForecasting,
    "realtime_trading_alpaca_paper": RealtimeStockTradingAlpacaPaper,
    "realtime_crypto_trading": RealtimeCryptoTrading,
    "realtime_crypto_forecasting": RealtimeCryptoForecasting,
    "offline_crypto_trading": OfflineCryptoTrading,
    "offline_crypto_forecasting": OfflineCryptoForecasting,
    "realtime_polymarket": RealtimePolymarket,
}


def create_app() -> FastAPI:
    spec = json.loads(CONFIG_PATH.read_text())
    if spec["kind"] not in TASKS:
        raise ValueError(f"Unsupported task kind {spec['kind']}")
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    task = TASKS[spec["kind"]](config=spec.get("config", {}))
    actions: list[Any] = []
    # Batch tasks score through /predict, so the verifier reads that result
    # rather than re-evaluating an action list it never filled.
    batch_score: dict = {}
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
        return batch_score or task.evaluate(actions)

    # Batch forecasting hands the agent every feature at once instead of
    # stepping, so it is served through its own pair of endpoints.
    @app.get("/features")
    def features() -> Any:
        return {
            "train_features": as_json(task.get_train_features()),
            "train_ground_truth": as_json(task.get_train_ground_truth()),
            "features": as_json(task.get_features()),
        }

    @app.post("/predict")
    def predict(payload: dict) -> dict:
        try:
            batch_score.update(
                task.predict_and_evaluate(as_arrays(payload["predictions"]))
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return batch_score

    return app
