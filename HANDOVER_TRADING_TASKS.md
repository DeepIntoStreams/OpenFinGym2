# Offline / realtime trading and forecasting tasks — handover

Untracked working note. Not for commit.
Branch: `realtime-trading-task` (worktree `/home/wenge/OpenFinGym2-realtime`), 16 commits ahead of `main`.

## Goal

Reproduce, inside OpenFinGym2's Harbor task format, the four curated task
families that already exist in the older OpenFinAIGym repository:

| family | agent does | data |
|---|---|---|
| `offline_trading` | places orders bar by bar | historical replay |
| `offline_forecasting` | predicts future values | historical replay |
| `realtime_trading` | places orders in the live market | live feed |
| `realtime_forecasting` | predicts, scored when the horizon elapses | live feed |

Two constraints set by the repository owner:

- **Behaviour must match the original**, including the same WebSocket, Binance
  and Alpaca setup. Not a reimplementation.
- **Replay is allowed only in the offline families.** The realtime families must
  use live data; no replay code in that path.

The new repository must be self-contained: nothing may depend on the older
repository or on an image published from it.

## Source material

The older repository is a separate checkout at `/home/wenge/OpenFinAIGym`
(remote `DeepIntoStreams/OpenFinAIGym`, HTTPS remote has no usable credentials
here — fetch the SSH URL explicitly). Its README describes the project but cites
no paper for itself. Per-task paper links live in
`tasks/curated_task_paper_matrix.md`, which covers only the 61 pipeline-generated
tasks; the 10 curated task bundles are hand-written and appear in no paper matrix.

That repository has local uncommitted deviations, documented in
`/home/wenge/OpenFinAIGym/LOCAL_DEVIATIONS.md`.

## What exists now

### Ported code — 12,053 lines under `src/open_fin_gym/realtime/`

Taken from the older repository and kept byte-faithful apart from import paths.
`ruff.toml` excludes this tree from linting for that reason (`force-exclude` is
required, since pre-commit passes explicit paths).

```
data_providers/   alpaca.py (REST + wss://stream.data.alpaca.markets)
                  binance.py (wss://stream.binance.com, @aggTrade + @depth)
                  base.py
execution/        simulated.py (GTC/IOC across bars, 1x buying power,
                                pessimistic fills), alpaca_paper.py, base.py
tasks/            base_realtime_task.py  (RealtimeForecastingTask)
                  realtime_trading_task.py
                  realtime_stock_trading.py     offline_stock_trading.py
                  realtime_stock_forecasting.py offline_stock_forecasting.py
contracts.py  offline_trading.py  market_data_buffer.py  ledger.py
rewards/  features.py  config.py
```

Two deliberate trims: `settings.py` + `schemas.py` (807 lines of config loading)
were replaced by `config.py` holding only the five-field `TradingConfig`, and
Polymarket was left out of this batch.

### Our own code — 76 lines under `src/open_fin_gym/broker/`

`server.py` exposes the task objects over HTTP so they can run as a compose
sidecar:

```python
TASKS = {"realtime_trading": RealtimeStockTrading,
         "realtime_forecasting": RealtimeStockForecasting,
         "offline_trading": OfflineStockTrading,
         "offline_forecasting": OfflineStockForecasting}
task = TASKS[spec["kind"]](config=spec.get("config", {}))
```

`/reset`, `/step`, `/score` map onto `task.reset()`, `task.step(action)` and
`task.evaluate(actions)`. `episode.json`'s `config` block corresponds exactly to
the older repository's `[curated.default_config]`.

### Image — `docker/broker.Dockerfile`

`python:3.12-slim` plus numpy, pandas, scipy, scikit-learn, torch 2.11,
pydantic, requests, websockets, fastapi, uvicorn. Build from the repository root:

```
docker build -t openfingym-broker -f docker/broker.Dockerfile .
```

`ksig` and `signatory` are guarded by try/except in `rewards/reward_bank.py`;
they are only needed by generation metrics and are absent by design.

### Pipeline integration

- `TaskType.TRADING` added, with `CURATED_TASK_TYPES` / `GENERATED_TASK_TYPES`.
  The two import-time exhaustiveness assertions now check `GENERATED_TASK_TYPES`,
  so trading is exempt from the generator and critic without weakening them.
- A `realtime_trading` scope exists in `conf/pipeline_config.yaml`, currently
  `enabled: false`.
- A `task_routing` step turns an accepted paper into a configured bundle: the LLM
  fills five fields from `descriptor.toml`'s schema and returns a `fit` verdict of
  `routed` / `partial_match` / `no_match`. A `no_match` writes no bundle.
- Papers routed this way get `PaperStatus.ROUTED_TO_CURATED` and skip extraction,
  critic and generation.

## Test status

### Passed

**Ported `realtime_stock_trading`, live market, direct container.** Five-step
episode (buy / hold ×3 / sell) on SPY. `evaluate()` returned the original nine
metrics. Slippage checked arithmetically: 5 shares × 772.07 × `slippage_pct=0.001`
= 3.86 per side, 7.72 round trip, matching `total_pnl` exactly. The WebSocket
layer logged its own warmup-timeout and REST-fallback message, i.e. the original
data path is live.

**All 12,053 lines import** inside the image, 9/9 modules.

**Harbor end-to-end**, with `verifier_result` produced and no exception:
compose brought up both services, `[environment.env]` injected the Alpaca
credentials, the oracle agent ran `solution/solve.sh`, real orders filled, and
the verifier wrote `reward.json` in `shared` mode. Repeated successfully against
the ported stack and the `openfingym-broker` image, with `verifier_result`
present and no exception: `pnl -0.2118`, `win_rate 0.2`, `num_trades 5`, matching
what the same episode produces when the broker is driven directly.

**Pipeline through routing**: 4 papers scraped → 1 accepted → 1 routed → bundle
exported, with the critic and generator correctly seeing zero candidates.

**`task.toml` schema**: validated against Harbor's `TaskConfig`, both for this
bundle and for mainline generated tasks.

### Not done

- `realtime_stock_trading_alpaca_paper` — the only task never run. It dispatches
  real orders to the Alpaca paper account, so it needs a decision on
  `flatten_on_start` first: the bundle currently sets `true`, which clears
  residual positions and open orders at construction, while the original default
  is to hard-fail on a dirty account without touching it.
- Sustained WebSocket operation — only the warmup path has been observed.
- The routing LLM has produced one configuration and chose every default, so a
  paper changing the output is still unproven.
- Harbor coverage is six of ten bundles: `offline_stock_trading`,
  `offline_crypto_trading`, `offline_stock_forecasting`,
  `offline_crypto_forecasting`, `realtime_crypto_trading` and
  `realtime_polymarket` all produced a `verifier_result` with no exception.
  `realtime_crypto_forecasting` reaches artifact collection and then trips the
  podman `cp` limitation below; the task itself runs correctly when driven
  directly. The three remaining bundles need the US equity session.
- The batch-forecasting reference solutions score poorly (`mape` around 1.0)
  because they predict the last feature column, which is a derived momentum
  value rather than a price. The task predicts an absolute close and exposes
  only derived features, and the agent image carries no numpy, so a sensible
  baseline is the mean of the training targets. Not yet changed.
- The routing LLM has produced only one configuration, and it chose every default
  value, so it is still unproven that a paper actually changes the output. Testing
  it against a paper with a clearly different market setup is the obvious next
  step and costs one call.

### Blocked by this machine, not by the code

`docker` here is podman 5.6.0-dev behind a shim. Three Harbor features cannot be
exercised as a result:

1. Platform detection — podman's version template has no `.Server.Arch`.
   Worked around locally with a shim that answers that one probe.
2. `network_mode = "allowlist"` — Harbor's egress-control sidecar needs buildx's
   `--output type=docker`, which podman build rejects. Tested with `public`
   instead; the committed `task.toml` keeps `allowlist`.
3. Sidecar artifact collection — podman-compose's `cp` parses the destination
   path as the service name. Harbor records the artifact as failed but does not
   fail the trial.

Also: podman-compose resolves a relative `build.context` against the overlay
compose file's directory rather than `--project-directory`, so the local Harbor
run used prebuilt images referenced by `image:` instead of `build:`.

None of these workarounds are committed, per instruction. Real Docker removes all
of them.

## Things to be careful about

- `AlpacaProvider` reads `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. The credentials
  in `/home/wenge/Task-Generation-Private/.env` are named `ALPACA_API_KEY_ID` and
  `ALPACA_API_SECRET_KEY`, so the bundle's `[environment.env]` has to map them.
- `artifacts` must sit above the first `[table]` header in `task.toml`. Placed
  lower it silently becomes `environment.artifacts` and is dropped — schema
  validation still passes.
- The original `step()` does not sleep. The agent may step as fast as it likes;
  `data_resolution` only governs buffering.
- `mlflow.langchain.autolog()` is called unconditionally in `__main__.py`, so a
  local pipeline run without a tracking server on :8080 stalls in retries. Use
  `~mlflow` to drop it. Left unfixed deliberately.
- The US equity session is the only window in which realtime tasks can run;
  outside it the task raises and the broker returns 503.
