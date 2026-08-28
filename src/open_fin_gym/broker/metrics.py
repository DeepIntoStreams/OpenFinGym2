import json
import math
from pathlib import Path


def score(ledger_path: Path) -> dict[str, float]:
    """
    Score a completed episode from its ledger

    Args:
        ledger_path: File written by the broker, one JSON record per step

    Returns:
        Metric dictionary with pnl as the headline value
    """
    records = [
        json.loads(line)
        for line in ledger_path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        return {
            "pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "steps": 0.0,
        }

    rewards = [float(r["reward"]) for r in records]
    equity = [float(r["equity"]) for r in records]
    opening = equity[0] - rewards[0]

    mean = sum(rewards) / len(rewards)
    variance = sum((x - mean) ** 2 for x in rewards) / len(rewards)
    sharpe = mean / math.sqrt(variance) if variance > 0 else 0.0

    peak, drawdown = opening, 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak if peak else 0.0)

    traded = [r for r in records if r.get("fill") and "id" in r["fill"]]
    wins = [r for r in traded if float(r["reward"]) > 0]

    return {
        "pnl": sum(rewards),
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        "win_rate": len(wins) / len(traded) if traded else 0.0,
        "total_return": (equity[-1] - opening) / opening if opening else 0.0,
        "steps": float(len(records)),
    }
