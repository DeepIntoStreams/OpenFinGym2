import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .broker import Broker, EpisodeConfig, MarketClosedError
from .metrics import score

LEDGER_PATH = Path(os.environ.get("BROKER_LEDGER", "/ledger/episode.jsonl"))
CONFIG_PATH = Path(os.environ.get("BROKER_CONFIG", "/broker/episode.json"))


def create_app() -> FastAPI:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    broker = Broker(EpisodeConfig(**json.loads(CONFIG_PATH.read_text())), LEDGER_PATH)
    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/reset")
    def reset() -> dict:
        try:
            return broker.reset()
        except MarketClosedError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/step")
    def step(action: dict) -> dict:
        try:
            return broker.step(action)
        except MarketClosedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/score")
    def episode_score() -> dict:
        return score(LEDGER_PATH)

    return app


app = create_app()
