import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import Paper


class Resolution(BaseModel):
    interval: str = Field(description="Bar interval, one of 1m, 5m, 15m, 1h")
    bars: int = Field(description="Number of bars of history at this interval")


class BundleConfig(BaseModel):
    reasoning: str
    symbols: list[str] = Field(description="Tickers the agent may trade")
    target_symbols: list[str] = Field(description="Subset of symbols the metrics score")
    data_resolution: str = Field(description="Bar cadence the episode steps at")
    context_resolutions: list[Resolution] = Field(
        description="Observation history per interval"
    )
    max_steps: int = Field(description="Bars per episode")


def find_bundles(bundles_path: Path, scope_id: str) -> list[tuple[Path, dict]]:
    """
    Find the curated bundles a scope routes onto

    Args:
        bundles_path: Directory holding curated bundles
        scope_id: Scope being routed

    Returns:
        List of (bundle directory, parsed descriptor) pairs
    """
    found = []
    for descriptor_path in sorted(bundles_path.glob("*/descriptor.toml")):
        descriptor = tomllib.loads(descriptor_path.read_text())
        if descriptor.get("scope_id") == scope_id:
            found.append((descriptor_path.parent, descriptor))
    return found


def build_routing_prompt(
    scope: Scope, paper: Paper, descriptor: dict, excerpt: str
) -> str:
    fields = "\n".join(
        f"- {name}: {spec.get('type')}. {spec.get('notes', '')}"
        + (f" Options: {spec['options']}." if "options" in spec else "")
        for name, spec in descriptor["config_schema"].items()
    )
    return f"""
You are configuring an existing benchmark task to match the market setup of a paper.

The task is fixed and is described as:

{descriptor["description"]}

Choose values for the following fields so the task reflects the instruments, frequency,
and horizon the paper trades. Where the paper is silent or its choice is unavailable,
keep the default rather than inventing something.

{fields}

Defaults:
{descriptor["defaults"]}

Constraints:
- The market is live US equities, so only tickers Alpaca serves are usable. Map a
  non-US or unavailable instrument onto the closest US-listed equivalent.
- data_resolution must be the finest interval in context_resolutions.
- An episode must fit inside one 6.5 hour trading session.

Scope: {scope.name}

Paper title: {paper.title}

{excerpt}
""".strip()
