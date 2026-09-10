from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from open_fin_gym.realtime.rewards import (
    MaxDrawdown,
    PnL,
    SharpeRatio,
    TradingReward,
    WinRate,
)


def _import_bundled_loader(load_path: Path) -> Any:
    """Import the per-task ``load.py`` deterministically from an absolute path.

    Phase 4's loader codegen stages a task-aware ``load.py`` next to the
    installed ``task.py`` (sibling files inside the Harbor environment's
    ``data/`` directory). We import it via ``importlib.util`` rather than
    a top-level ``import load`` so multiple tasks in the same process do
    not collide on ``sys.modules["load"]``. Each call uses a unique
    module name suffixed with the path's hash.

    This helper lives in framework code (contracts.py) on purpose: the
    LLM-generated task code is screened by an AST sandbox that blocks
    ``importlib`` (see ``utils/sandbox.py``), so any dynamic import of
    ``load.py`` must happen inside a base class the user code merely
    inherits.
    """
    import importlib.util
    import sys

    module_name = f"_dataset_loader_{abs(hash(str(load_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(load_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {load_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_task_data_dir(task_obj: "BaseTask") -> Path:
    """Locate the directory holding the task's bundled ``load.py``.

    Priority: ``config["data_dir"]`` (explicit override, used by tests
    and custom installs) -> the directory containing the subclass's
    module file (the standard Harbor layout where ``task.py``,
    ``load.py``, and the dataset payload live side by side).

    Raising here is the right behaviour: a missing data dir means the
    task package is broken, and the gym-loop fallbacks downstream
    (``reset`` calling ``load_data``) cannot recover silently.
    """
    import sys

    cfg_dir = (
        task_obj.config.get("data_dir") if isinstance(task_obj.config, dict) else None
    )
    if cfg_dir:
        candidate = Path(cfg_dir).resolve()
        if candidate.is_dir():
            return candidate

    cls_module = sys.modules.get(type(task_obj).__module__)
    module_file = getattr(cls_module, "__file__", None) if cls_module else None
    if module_file is None:
        raise RuntimeError(
            f"cannot resolve data directory for {type(task_obj).__name__}: "
            "the subclass module has no __file__ attribute. Pass "
            "config={'data_dir': '/path/to/dir'} when instantiating."
        )
    return Path(module_file).resolve().parent


@dataclass
class TaskMetadata:
    task_id: str
    title: str
    description: str
    task_type: str = "offline"  # "offline" | "realtime"
    interaction_model: str = "gym"  # "forecasting" | "trading" | "gym" (legacy default)
    source_papers: list[str] | None = None
    tags: list[str] | None = None
    data_requirements: list[str] | None = None
    difficulty: str = "medium"
    version: str = "1.0.0"


class BaseTask(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._data: Any = None

    @abstractmethod
    def metadata(self) -> TaskMetadata:
        raise NotImplementedError

    def load_data(self) -> Any:
        """Default loader: import the bundled per-task ``load.py``.

        Phase 4's loader codegen stages a task-aware ``load.py`` next
        to the installed ``task.py`` (sibling files inside the Harbor
        environment's ``data/`` directory). The default ``load_data``
        finds it via the subclass's module file, imports it
        deterministically, calls ``load(data_dir)``, and caches the
        resulting ``{"features": ..., "ground_truth": ...}`` dict on
        ``self._data``.

        Subclasses override only when their data flow does not match
        this pattern (e.g. realtime tasks that stream from a market
        feed instead of a static dataset). LLM-authored task packages
        should leave this method alone — the AST sandbox blocks
        ``importlib`` in user code, so attempts to reimplement this
        body in ``task.py`` will fail static analysis.
        """
        if self._data is not None:
            return self._data
        data_dir = _resolve_task_data_dir(self)
        load_py = data_dir / "load.py"
        if not load_py.exists():
            raise FileNotFoundError(
                f"bundled load.py not found at {load_py}. Phase 4 must "
                "generate and install a per-task loader for this task "
                "(see openfinai_pipeline.benchmark.loader). Subclasses "
                "with bespoke data flows should override load_data."
            )
        module = _import_bundled_loader(load_py)
        loader_fn = getattr(module, "load", None)
        if not callable(loader_fn):
            raise ImportError(
                f"{load_py} does not expose a callable load(data_dir) "
                "function — regenerate the per-task loader."
            )
        self._data = loader_fn(data_dir)
        return self._data

    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError

    def prepare_eval_context(
        self,
        predictions: Any,
        ground_truth: Any,
    ) -> Dict[str, Any]:
        """Optional hook: return extra context keys for the evaluator.

        Default is empty. Tasks whose evaluator includes an
        ``embedding_pair`` metric (e.g. ``EmbeddingFID``) override this to
        return ``{"real_emb": ..., "fake_emb": ...}`` — the feature
        extractor runs upstream so reward metrics stay model-free.
        Returned keys are merged into the ``**kwargs`` passed to the
        evaluator's ``score`` call via
        :meth:`ForecastingTask.predict_and_evaluate` /
        :meth:`GenerativeTask.generate_and_evaluate`.
        """
        return {}


class ForecastingTask(BaseTask):
    """Task where the agent produces predictions scored against ground truth.

    Forecasting has a fundamentally different interaction model from trading:
    the agent produces predictions and the result is scored against ground
    truth (which may be immediate for offline batch tasks or deferred for
    realtime streaming tasks).

    Two sub-patterns exist, controlled by :attr:`batch_mode`:

    * ``batch_mode = True`` (default) -- **Offline batch forecasting**.
      The agent receives the entire feature set in a single ``act()`` call
      and returns all predictions at once.  The batch path bypasses the
      gym loop and calls :meth:`predict_and_evaluate` directly.
      Natural for historical datasets (ACL18, KDD17, Yahoo Finance).

    * ``batch_mode = False`` -- **Streaming forecasting**.
      The agent makes one prediction per ``step()`` call, each potentially
      using fresh data (e.g. real-time market snapshots).  Predictions are
      independent and ground truth may be deferred.  The runner uses the
      standard gym loop.  Subclasses typically override ``reset``, ``step``,
      and ``evaluate`` to handle their custom data source.

    Subclasses must implement the usual :class:`BaseTask` abstract methods
    (``metadata``, ``load_data``, ``get_observation_space``, ``get_action_space``)
    plus :meth:`get_features` -- the feature set used for prediction.

    Train accessors (``get_train_features`` / ``get_train_ground_truth``)
    are concrete with defaults that read from the B-shape ``self._data``
    bundle the auto-pipeline installs. Curated tasks may override.

    :meth:`get_ground_truth` exists for back-compat with curated tasks
    (which return ``y_test`` directly). Auto-generated tasks DO NOT
    override it; the default implementation raises ``PermissionError``
    so an agent-side caller cannot accidentally read the held-out test
    target through the framework. The verifier reads test ground truth
    from a separate ``/eval-data/test_ground_truth.h5`` artifact via
    the assembled evaluator's ``_load_reference_data``.
    """

    # Runner dispatch flag; streaming subclasses (RealtimeForecastingTask)
    # set False so the runner uses the standard gym step loop.
    batch_mode: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._evaluator: Optional["BaseEvaluator"] = None
        self._current_idx: int = 0
        self._done: bool = False
        self._predictions: List[Any] = []

    @abstractmethod
    def get_features(self) -> Any:
        """Return the feature set the agent uses for prediction.

        For auto-pipeline tasks this returns the **test** features only
        (the agent's predictions on these are scored by the verifier).
        For curated tasks it may return whatever the hand-written class
        chooses.
        """
        raise NotImplementedError

    def get_train_features(self) -> Any:
        """Training features for the agent to fit a model on.

        Default impl reads ``self._data["train"]["features"]`` (the
        B-shape bundle the auto-pipeline installs). Curated tasks
        override when their data flow does not match.
        """
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict) or "train" not in self._data:
            raise NotImplementedError(
                "default get_train_features expects self._data to be the "
                "B-shape bundle {'train': {...}, 'test': {...}}; this task "
                "should override get_train_features"
            )
        return self._data["train"]["features"]

    def get_train_ground_truth(self) -> Any:
        """Training ground-truth labels for the agent to fit against.

        Default impl reads ``self._data["train"]["ground_truth"]``. The
        agent uses this to compute their training loss; nothing here
        leaks the held-out test target.
        """
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict) or "train" not in self._data:
            raise NotImplementedError(
                "default get_train_ground_truth expects self._data to be "
                "the B-shape bundle {'train': {...}, 'test': {...}}; this "
                "task should override get_train_ground_truth"
            )
        return self._data["train"]["ground_truth"]

    def get_ground_truth(self) -> Any:
        """**DO NOT CALL FROM AGENT CODE.** Held-out test target.

        For auto-pipeline tasks, calling this raises ``PermissionError``
        — the test ground_truth is intentionally held out from the
        agent and the framework's ``BaseTask.load_data`` populates
        ``self._data["test"]["ground_truth"]`` with ``None`` to enforce
        the contract. The verifier reads the real test target from
        ``/eval-data/test_ground_truth.h5`` via the assembled
        evaluator's ``_load_reference_data``.

        Curated tasks (e.g. the hand-written ``OfflineStockForecasting``
        class under ``tasks/offline_stock_forecasting/``) override this
        method to return ``y_test`` directly. That's safe in the
        curated path because curated tasks have their own
        evaluation entry point and don't go through the
        ``predict_and_evaluate(split=)`` gate.
        """
        raise PermissionError(
            "ForecastingTask.get_ground_truth() must not be called from "
            "agent-side code: the held-out test target is reserved for "
            "the verifier. Use get_train_ground_truth() to fit a model. "
            "If you are writing a curated task, override this method on "
            "your subclass."
        )

    def predict_and_evaluate(
        self,
        predictions: Any,
        *,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Score ``predictions`` against ``split``'s ground truth.

        ``split`` semantics:

        * ``"train"`` -- score against ``self.get_train_ground_truth()``.
          The agent uses this to sanity-check their model on training
          data using the curated reward bank.
        * ``"test"`` -- raises :class:`PermissionError`. Test scoring
          is verifier-only.
        * ``None`` (legacy) -- calls ``self.get_ground_truth()``. For
          auto-pipeline tasks this raises ``PermissionError``; for
          curated tasks that override ``get_ground_truth``, it falls
          through to the historical "score against y_test" behaviour.

        ``kwargs["reward_output"]``, when provided, MUST NOT point at
        ``/logs/verifier/`` -- that is Harbor's canonical reward channel
        and only the verifier writes there. We guard against accidental
        pollution explicitly.
        """
        if split == "test":
            raise PermissionError(
                "split='test' is verifier-only; use split='train' for "
                "agent-side self-evaluation"
            )
        reward_output = kwargs.get("reward_output")
        if reward_output is not None:
            ro_str = str(reward_output)
            if "/logs/verifier/" in ro_str.replace("\\", "/"):
                raise PermissionError(
                    "predict_and_evaluate cannot write into /logs/verifier/ "
                    "— that path is reserved for the verifier's canonical "
                    "reward.json. Pick a path under your workspace."
                )

        if split is None:
            ground_truth = self.get_ground_truth()
        elif split == "train":
            ground_truth = self.get_train_ground_truth()
        else:
            raise ValueError(
                f"unknown split {split!r}; valid: 'train' (agent-side) "
                "or omit for legacy curated-task behaviour"
            )

        if self._evaluator is not None:
            extra_ctx = self.prepare_eval_context(predictions, ground_truth) or {}
            merged = {**extra_ctx, **kwargs}  # explicit kwargs win on collision
            return self._evaluator.score(predictions, ground_truth, **merged)
        return self._default_score(predictions, ground_truth)

    # ------------------------------------------------------------------
    def reset(self) -> Any:
        if self._data is None:
            self.load_data()
        self._current_idx = 0
        self._done = False
        self._predictions = []
        return self._get_observation_at(0)

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        """Compatibility shim: record prediction, advance index.

        Forecasting tasks accumulate predictions; the real scoring happens
        in :meth:`evaluate` / :meth:`predict_and_evaluate`. Per-step reward
        is always ``0.0`` -- evaluation is inherently batch/deferred.
        """
        if self._done:
            raise RuntimeError(
                "step() called on a finished episode. Call reset() first."
            )
        self._predictions.append(action)
        self._current_idx += 1
        total = self._get_num_samples()
        self._done = self._current_idx >= total
        obs_idx = min(self._current_idx, max(total - 1, 0))
        obs = self._get_observation_at(obs_idx)
        return obs, 0.0, self._done, {"prediction_idx": self._current_idx - 1}

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Score ``agent_actions`` against ground truth via the evaluator.

        Unlike the legacy trading-style ``evaluate`` which replays the
        episode, this delegates to :class:`BaseEvaluator.score` when one
        is wired, or falls back to a default scoring function.
        """
        return self.predict_and_evaluate(agent_actions, **kwargs)

    # ------------------------------------------------------------------
    def _get_num_samples(self) -> int:
        """Number of samples in the dataset. Default: ``len(features)``."""
        try:
            features = self.get_features()
        except Exception:
            return 1
        if features is None:
            return 1
        if hasattr(features, "__len__"):
            try:
                return int(len(features))
            except TypeError:
                pass
        return 1

    def _get_observation_at(self, idx: int) -> Any:
        """Observation at index ``idx``. Default: ``features[idx]``."""
        try:
            features = self.get_features()
        except Exception:
            return None
        if features is None:
            return None
        if hasattr(features, "__getitem__"):
            try:
                return features[idx]
            except (IndexError, KeyError):
                return None
        return features

    def _default_score(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Fallback scorer used when no evaluator is wired.

        Computes directional accuracy on flattened 1-D arrays. Returns an
        empty result when either input is missing.
        """
        if predictions is None or ground_truth is None:
            return {"directional_accuracy": 0.0}
        try:
            import numpy as np

            pred = np.asarray(predictions).astype(float).flatten()
            gt = np.asarray(ground_truth).astype(float).flatten()
        except Exception:
            return {"directional_accuracy": 0.0}
        n = min(len(pred), len(gt))
        if n == 0:
            return {"directional_accuracy": 0.0}
        pred, gt = pred[:n], gt[:n]
        correct = float(((pred > 0) == (gt > 0)).sum())
        return {"directional_accuracy": correct / n}


class GenerativeTask(BaseTask):
    """Task where the agent generates samples scored against reference data.

    Two sub-shapes are produced by the auto-pipeline:

    * **Conditional generative** -- the LLM loader emits a B-shape bundle
      ``{"train": {features, ground_truth}, "test": {features, ground_truth}}``
      where ``features`` is conditioning and ``ground_truth`` is a real
      reference sample. The verifier scores the agent's generated samples
      against the held-out test reference; the agent fits using
      :meth:`get_train_features` (train conditioning) and
      :meth:`get_train_reference_data` (train real samples).

    * **Unconditional generative** (``split_policy="no_split"``) -- the
      LLM emits ``{"reference": ndarray}`` with no held-out target.
      Distributional metrics like FID/KID compare the agent's
      generated-sample distribution to the full reference distribution;
      :meth:`get_reference_data` returns the full reference (agent and
      verifier see the same data because there is no held-out
      distinction).

    Curated tasks override the legacy :meth:`get_reference_data` to
    return the real reference directly. Auto-pipeline tasks DO NOT
    override it for the conditional case (the default raises so the
    held-out test reference is not exposed); they DO override it for
    the unconditional case to point at ``self._data["reference"]``.
    """

    batch_mode: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._evaluator: Optional["BaseEvaluator"] = None
        self._generated_samples: list[Any] = []

    def get_reference_data(self) -> Any:
        """**Held-out test reference** (conditional generative).

        Default implementation raises ``PermissionError`` for the same
        reason as :meth:`ForecastingTask.get_ground_truth`: the test
        reference is verifier-only in the auto-pipeline. Curated tasks
        and unconditional-generative auto-pipeline tasks override this
        to return the appropriate reference set.
        """
        raise PermissionError(
            "GenerativeTask.get_reference_data() must not be called from "
            "agent-side code in the conditional case: the held-out test "
            "reference is reserved for the verifier. Use "
            "get_train_reference_data() to fit a generator. Unconditional "
            "generative tasks override this method on their subclass."
        )

    def get_train_features(self) -> Any:
        """Training conditioning inputs (or training reference for unconditional)."""
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict):
            raise NotImplementedError(
                "default get_train_features expects self._data to be a "
                "B-shape or reference-shape bundle dict"
            )
        if "train" in self._data:
            return self._data["train"]["features"]
        if "reference" in self._data:
            # Unconditional generative: agent is allowed to see the full
            # reference distribution, so returning it as features is valid.
            return self._data["reference"]
        raise NotImplementedError(
            "default get_train_features cannot find a 'train' bundle or "
            "'reference' key in self._data"
        )

    def get_train_reference_data(self) -> Any:
        """Training real reference samples for the agent to fit a generator on."""
        if self._data is None:
            self.load_data()
        if not isinstance(self._data, dict):
            raise NotImplementedError(
                "default get_train_reference_data expects self._data to be a "
                "B-shape or reference-shape bundle dict"
            )
        if "train" in self._data:
            return self._data["train"]["ground_truth"]
        if "reference" in self._data:
            return self._data["reference"]
        raise NotImplementedError(
            "default get_train_reference_data cannot find a 'train' bundle "
            "or 'reference' key in self._data"
        )

    def get_conditioning_data(self) -> Any:
        """Return optional conditioning inputs for conditional generation."""
        return None

    def generate_and_evaluate(
        self,
        generated_samples: Any,
        *,
        split: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Score generated samples against reference. See ``ForecastingTask.predict_and_evaluate`` for split semantics."""
        if split == "test":
            raise PermissionError(
                "split='test' is verifier-only; use split='train' for "
                "agent-side self-evaluation"
            )
        reward_output = kwargs.get("reward_output")
        if reward_output is not None:
            ro_str = str(reward_output)
            if "/logs/verifier/" in ro_str.replace("\\", "/"):
                raise PermissionError(
                    "generate_and_evaluate cannot write into /logs/verifier/ "
                    "— that path is reserved for the verifier's canonical "
                    "reward.json. Pick a path under your workspace."
                )

        if split is None:
            reference_data = self.get_reference_data()
        elif split == "train":
            reference_data = self.get_train_reference_data()
        else:
            raise ValueError(
                f"unknown split {split!r}; valid: 'train' (agent-side) "
                "or omit for legacy curated-task behaviour"
            )

        if self._evaluator is not None:
            extra_ctx = (
                self.prepare_eval_context(generated_samples, reference_data) or {}
            )
            merged = {**extra_ctx, **kwargs}  # explicit kwargs win on collision
            return self._evaluator.score(generated_samples, reference_data, **merged)
        return self._default_score(generated_samples, reference_data)

    def reset(self) -> Any:
        if self._data is None:
            self.load_data()
        self._generated_samples = []
        conditioning = self.get_conditioning_data()
        if conditioning is not None:
            return conditioning
        # Try train-side accessor (auto-pipeline), then legacy get_reference_data.
        try:
            return self.get_train_reference_data()
        except (NotImplementedError, PermissionError):
            return self.get_reference_data()

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        self._generated_samples.append(action)
        return None, 0.0, True, {"generated_batches": len(self._generated_samples)}

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        if len(agent_actions) == 1:
            payload = agent_actions[0]
        else:
            payload = agent_actions
        return self.generate_and_evaluate(payload, **kwargs)

    def _default_score(
        self, generated_samples: Any, reference_data: Any
    ) -> Dict[str, float]:
        if generated_samples is None or reference_data is None:
            return {"reward": 0.0}
        try:
            import numpy as np

            fake = np.asarray(generated_samples, dtype=float)
            real = np.asarray(reference_data, dtype=float)
        except Exception:
            return {"reward": 0.0}

        if fake.size == 0 or real.size == 0:
            return {"reward": 0.0}

        fake_mean = float(fake.mean())
        real_mean = float(real.mean())
        mean_gap = abs(fake_mean - real_mean)
        return {"mean_gap": mean_gap, "reward": 1.0 / (1.0 + mean_gap)}


@dataclass
class PortfolioState:
    """Tracks a trading portfolio across steps.

    ``positions`` maps symbol -> quantity held (signed -- negative
    means short). ``history`` records a chronological log of state
    snapshots or trade events for auditing and metric computation.
    ``pending_orders`` holds open limit/stop/stop_limit orders awaiting
    a fill (only used in transactional offline mode); ``reserved_cash``
    is the cumulative cash earmarked by those buy orders so successive
    submissions can't double-spend.
    """

    cash: float = 10000.0
    positions: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    pending_orders: Dict[str, Any] = field(default_factory=dict)
    reserved_cash: float = 0.0

    def market_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Total portfolio value given current ``prices``.

        When ``prices`` is ``None``, positions are valued at zero (useful
        only for reporting cash-only balances).
        """
        prices = prices or {}
        holdings_value = sum(
            prices.get(symbol, 0.0) * qty for symbol, qty in self.positions.items()
        )
        return self.cash + holdings_value


@dataclass
class TradingAction:
    """Structured trading action with quantity support.

    Used by :class:`RealtimeTradingTask` for realtime paper trading.  Offline
    :class:`TradingTask` continues to accept plain ``int`` actions
    (``{-1, 0, 1}``) for backward compatibility.
    """

    action: str  # "buy" | "sell" | "hold"
    symbol: str
    quantity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradingAction":
        """Parse a dict (typically from agent output) into a TradingAction."""
        return cls(
            action=d["action"],
            symbol=d["symbol"],
            quantity=float(d.get("quantity", 0.0)),
            metadata={
                k: v for k, v in d.items() if k not in ("action", "symbol", "quantity")
            },
        )


# ---------------------------------------------------------------------------
# Order-dispatch glue shared by every trading task (offline replay +
# realtime live). Actions are transactional dicts (``{action, symbol,
# quantity, order_type, limit_price, stop_price, tif, ...}`` or
# ``{"orders": [...]}``); the executor matches market/limit/stop/stop_limit
# with IOC or GTC, plus cancel. Offline passes an OHLC bar quote (intrabar
# low/high triggering); realtime passes a scalar tick.
# ---------------------------------------------------------------------------


def dispatch_orders_via_executor(
    executor: Any,
    orders: List[Dict[str, Any]],
    quotes: Dict[str, Any],
    *,
    known_symbols: set,
    step: int,
    timestamp: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Process pending orders + dispatch a step's orders through a ``BaseExecutor``.

    ``quotes`` maps symbol -> quote, where a quote is either a scalar tick
    price (realtime) or an OHLC bar dict ``{open,high,low,close}`` (offline
    replay) — ``SimulatedExecutor`` normalises both. This is the single
    order-matching glue shared by the offline transactional path (and, in
    the unified base, realtime); it replaces the old per-task
    ``offline_submit_order`` / ``offline_process_pending_against_bars``
    duplication. Returns ``{"fills", "accepted", "rejections", "expired"}``
    as lists of dicts ready to drop into ``info``.
    """
    from open_fin_gym.realtime.execution import (
        ActionVerb,
        OrderIntent,
        OrderStatus,
        Rejection,
        RejectionCode,
    )

    fills: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    expired: List[Dict[str, Any]] = []

    # Phase 1: process pending orders queued from prior steps against the
    # current quotes (bar OHLC offline / latest tick realtime).
    tick_res = executor.tick(quotes, step=step, timestamp=timestamp)
    for f in tick_res.fills:
        fills.append(f.to_dict())
    for e in tick_res.expirations:
        expired.append(e.to_dict())

    # Phase 2: dispatch this step's orders (cancels + new submits).
    for od in orders:
        verb = od.get("action")
        if verb == ActionVerb.CANCEL:
            order_id = od.get("order_id")
            if not isinstance(order_id, str) or not order_id:
                rejections.append(
                    {
                        "order_intent": od,
                        "reason_code": RejectionCode.MISSING_FIELD,
                        "reason": "cancel requires a string order_id",
                    }
                )
                continue
            cres = executor.cancel(order_id)
            if cres.is_cancelled and cres.cancelled is not None:
                expired.append(
                    {**cres.cancelled.to_dict(), "status": OrderStatus.CANCELLED}
                )
            elif cres.rejection is not None:
                rejections.append(cres.rejection.to_dict())
            continue

        symbol = od.get("symbol")
        if (
            verb in (ActionVerb.BUY, ActionVerb.SELL)
            and isinstance(symbol, str)
            and symbol not in known_symbols
        ):
            rejections.append(
                {
                    "order_intent": od,
                    "reason_code": RejectionCode.UNKNOWN_SYMBOL,
                    "reason": (
                        f"symbol {symbol!r} is not in this task's "
                        f"symbols={sorted(known_symbols)}"
                    ),
                }
            )
            continue

        parsed = OrderIntent.from_dict(od)
        if isinstance(parsed, Rejection):
            rejections.append(parsed.to_dict())
            continue

        quote = quotes.get(parsed.symbol)
        if quote is None:
            rejections.append(
                {
                    "order_intent": od,
                    "reason_code": RejectionCode.UNKNOWN_SYMBOL,
                    "reason": f"no quote for symbol {parsed.symbol!r}",
                }
            )
            continue

        sub = executor.submit(
            parsed, market_price=quote, step=step, timestamp=timestamp
        )
        if sub.is_filled and sub.fill is not None:
            fills.append(sub.fill.to_dict())
        elif sub.is_accepted and sub.accepted is not None:
            accepted.append(sub.accepted.to_dict())
        elif sub.is_rejected and sub.rejection is not None:
            rejections.append(sub.rejection.to_dict())

    return {
        "fills": fills,
        "accepted": accepted,
        "rejections": rejections,
        "expired": expired,
    }


def _trade_history_to_pairs(
    history: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Adapt a trade-history stream into :class:`TradingReward` input pairs.

    The per-step ``reward`` stored by :meth:`TradingTask.step` (or
    :meth:`RealtimeTradingTask.step`) is already signed by
    ``_execute_trade`` (e.g. ``position_delta * price_change`` offline,
    ``total_pnl - prev_pnl`` realtime). Fixing ``direction="long"`` and
    ``quantity=1.0`` makes :func:`PnL._pnl_list` degenerate to the
    identity on that signed stream — any attempt to re-derive direction
    from the action would double-count the sign on short losses.
    """
    predictions: List[Dict[str, Any]] = []
    ground_truths: List[Dict[str, Any]] = []
    for entry in history:
        predictions.append({"direction": "long", "quantity": 1.0})
        ground_truths.append({"actual_return": float(entry["reward"])})
    return predictions, ground_truths


def _compute_trading_metrics_from_history(
    history: List[Dict[str, Any]],
    reward_classes: Sequence[type[TradingReward]],
    *,
    count_label: str = "num_trades",
) -> Dict[str, float]:
    """Shared metric helper for :class:`TradingTask` + :class:`RealtimeTradingTask`.

    Scalar stats (``total_return``, ``<count_label>``, ``mean_return``,
    ``std_return``) are computed inline; the caller's ``reward_classes``
    supply everything else (PnL, Sharpe, drawdown, win-rate, ...) via the
    reward bank's ``compute_aggregate``. Each metric is keyed by its
    ``.name`` attribute.

    Per-fill / rejection / expiration audit rows (``kind`` keyed) are
    skipped so they don't double-count alongside per-step aggregates.
    """
    aggregate_history = [e for e in history if "kind" not in e]
    returns = [float(entry["reward"]) for entry in aggregate_history]
    metrics: Dict[str, float] = {
        "total_return": float(sum(returns)),
        count_label: float(len(returns)),
    }
    if not returns:
        return metrics

    import numpy as np

    arr = np.asarray(returns, dtype=float)
    metrics["mean_return"] = float(arr.mean())
    metrics["std_return"] = float(arr.std())

    predictions, ground_truths = _trade_history_to_pairs(aggregate_history)
    for cls in reward_classes:
        try:
            metric = cls()
            value = float(metric.compute_aggregate(predictions, ground_truths))
            metrics[metric.name] = value
        except Exception:
            # Fall back to class name when instantiation itself fails.
            metrics[cls.__name__.lower()] = float("nan")
    return metrics


class TradingTask(BaseTask):
    """Task where the agent makes sequential trading decisions.

    Trading has a fundamentally different interaction model from forecasting:
    the agent observes an evolving market, executes orders, and portfolio
    state carries between steps. The gym-style ``reset`` / ``step`` /
    ``evaluate`` loop IS the natural interface here.

    Subclasses must implement the standard :class:`BaseTask` abstracts
    (``metadata``, ``load_data``, ``get_observation_space``, ``get_action_space``)
    plus two new hooks:

    * :meth:`_get_market_observation` -- build the current observation
      (market state + portfolio state) used by the agent
    * :meth:`_execute_trade` -- translate the agent's action into a trade,
      return a per-step reward and an info dict

    ``evaluate`` reads metrics from the accumulated ``_trade_history``
    (NO episode replay).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._portfolio: PortfolioState = PortfolioState()
        self._step_count: int = 0
        self._done: bool = False
        self._trade_history: List[Dict[str, Any]] = []
        # Per-symbol cumulative-PnL watermark; the per-step reward is the
        # delta against it (reward[sym] = compute_pnl[sym] - prev). Shared
        # by the offline + realtime per-step reward path.
        self._prev_pnl_per_sym: Dict[str, float] = {}

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        return list(self._trade_history)

    # ------------------------------------------------------------------
    # Data-source seam — the ONLY per-task divergence (historical replay
    # vs live feed). Everything else (order dispatch, reward, observation
    # skeleton, evaluation) is shared below.
    # ------------------------------------------------------------------

    @abstractmethod
    def _current_prices(self) -> Dict[str, float]:
        """Latest scalar price per symbol (drives PnL + portfolio value)."""
        raise NotImplementedError

    @abstractmethod
    def _execution_quotes(self) -> Dict[str, Any]:
        """Quote per symbol fed to the executor.

        Offline returns an OHLC bar dict (``{open,high,low,close}``) so
        limit/stop orders fill intrabar; realtime returns a scalar tick
        (the in-progress bar's high/low aren't known yet).
        :meth:`SimulatedExecutor._quote_bounds` normalizes both.
        """
        raise NotImplementedError

    @abstractmethod
    def _market_observation_block(self) -> Dict[str, Dict[str, Any]]:
        """Per-symbol market sub-block of the observation.

        The one genuinely agent-visible difference between offline and
        realtime: offline carries closed-bar OHLCV + ``order_book: None``
        (historical data has no book); realtime carries live
        ``recent_bars`` + an ``order_book`` snapshot.
        """
        raise NotImplementedError

    @abstractmethod
    def _episode_done(self) -> bool:
        """Whether this step terminates the episode."""
        raise NotImplementedError

    @abstractmethod
    def _steps_remaining(self) -> int:
        """Steps left in the episode (sentinel ``-1`` if unbounded)."""
        raise NotImplementedError

    def _market_timestamp(self) -> Any:
        """Execution timestamp passed to the executor. Default ``None``."""
        return None

    def close(self) -> None:
        """Release task resources. Default no-op (realtime overrides)."""

    # ------------------------------------------------------------------
    # Shared per-step engine + observation + reward (offline == realtime)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_action(action: Any) -> List[Dict[str, Any]]:
        """Normalise an action into a list of single-order dicts.

        Accepts a :class:`TradingAction`, a single ``{"action": ...}``
        dict, a ``{"orders": [...]}`` batch, or a list/tuple of order
        dicts. Anything else surfaces as one malformed entry so the
        executor rejects it cleanly.
        """
        if isinstance(action, TradingAction):
            return [
                {
                    "action": action.action,
                    "symbol": action.symbol,
                    "quantity": action.quantity,
                    **action.metadata,
                }
            ]
        if isinstance(action, dict):
            if "orders" in action and isinstance(action["orders"], list):
                return [o for o in action["orders"] if isinstance(o, dict)]
            return [action]
        if isinstance(action, (list, tuple)):
            return [o for o in action if isinstance(o, dict)]
        return [{"action": "<malformed>", "raw": action}]

    def _execute_trade(self, action: Any) -> tuple[float, Dict[str, Any]]:
        """Tick pending orders, dispatch the agent's orders, score the step.

        Reward is the per-symbol ``compute_pnl`` delta against the previous
        step's watermark, summed across symbols — identical for offline and
        realtime. ``per_symbol_rewards`` is retained so ``evaluate`` can
        scope the headline metrics to ``target_symbols``.
        """
        orders = self._normalize_action(action)
        quotes = self._execution_quotes()
        prices = self._current_prices()
        step_index = self._step_count + 1
        ts = self._market_timestamp()

        disp = dispatch_orders_via_executor(
            self._executor,
            orders,
            quotes,
            known_symbols=set(self._symbols),
            step=step_index,
            timestamp=ts,
        )

        pnl = self._executor.compute_pnl(prices)
        per_symbol_rewards: Dict[str, float] = {}
        for sym in self._symbols:
            cur = float(pnl.get(sym, 0.0))
            per_symbol_rewards[sym] = cur - self._prev_pnl_per_sym.get(sym, 0.0)
            self._prev_pnl_per_sym[sym] = cur
        total_reward = float(sum(per_symbol_rewards.values()))

        return total_reward, {
            "done": self._episode_done(),
            "prices": prices,
            "total_pnl": float(pnl.get("__total", 0.0)),
            "per_symbol_rewards": per_symbol_rewards,
            "fills": disp["fills"],
            "accepted": disp["accepted"],
            "rejections": disp["rejections"],
            "expired": disp["expired"],
        }

    def _get_market_observation(self) -> Dict[str, Any]:
        """Unified observation: per-symbol market block + portfolio block.

        The portfolio block is read straight from the executor and is
        byte-identical across offline/realtime; the market block is the
        per-task seam (see :meth:`_market_observation_block`).
        """
        prices = self._current_prices()
        positions = self._executor.get_positions()
        cash = self._executor.get_cash()
        value: Optional[float]
        if cash is None:
            value = None
        else:
            value = float(cash) + sum(
                float(qty) * prices.get(sym, 0.0) for sym, qty in positions.items()
            )
        return {
            "step": self._step_count,
            "steps_remaining": self._steps_remaining(),
            "symbols": self._market_observation_block(),
            "portfolio": {
                "cash": cash,
                "reserved_cash": self._executor.get_reserved_cash(),
                "positions": positions,
                "pnl": self._executor.compute_pnl(prices),
                "value": value,
                "pending_orders": [
                    o.to_dict() for o in self._executor.get_pending_orders()
                ],
            },
        }

    # ------------------------------------------------------------------
    # BaseTask contract -- concrete implementations
    # ------------------------------------------------------------------

    def reset(self) -> Any:
        initial_cash = float(self.config.get("initial_cash", 10000.0))
        self._portfolio = PortfolioState(cash=initial_cash)
        self._step_count = 0
        self._done = False
        self._trade_history = []
        self._prev_pnl_per_sym = {}
        if self._data is None:
            self.load_data()
        return self._get_market_observation()

    def _sync_portfolio_from_executor(self) -> None:
        """Mirror the executor's cash/positions/pending into PortfolioState.

        The transactional path runs order matching on ``self._executor``
        (a :class:`SimulatedExecutor`); the observation reads
        :class:`PortfolioState`, so we sync after each transactional step.
        No-op when the subclass doesn't hold an executor (legacy / stub).
        """
        ex = getattr(self, "_executor", None)
        if ex is None:
            return
        self._portfolio.cash = float(ex.get_cash() or 0.0)
        self._portfolio.positions = dict(ex.get_positions())
        self._portfolio.reserved_cash = float(ex.get_reserved_cash())
        self._portfolio.pending_orders = {
            o.order_id: o for o in ex.get_pending_orders()
        }

    def _on_step_start(self) -> None:
        """Hook at the top of :meth:`step`, before the trade executes.

        Default no-op (offline replay needs nothing — the cursor advances
        implicitly via ``_step_count``). :class:`RealtimeTradingTask`
        overrides it to refresh the live market buffer so the reward and
        the next observation reflect the freshest prices.
        """

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError(
                "step() called on a finished episode. Call reset() first."
            )
        self._on_step_start()
        reward, info = self._execute_trade(action)
        info = dict(info)  # copy to avoid mutating caller-supplied dicts
        self._step_count += 1
        self._trade_history.append(
            {
                "step": self._step_count,
                "action": action,
                "reward": float(reward),
                **info,
            }
        )
        # Per-fill / rejection / expiration audit rows (reward=0 so they
        # don't double-count in the metric stream; evaluate() / the
        # reward-bank helper skip ``kind``-tagged entries).
        for fill in info.get("fills", []):
            self._trade_history.append(
                {"step": self._step_count, "kind": "fill", **fill, "reward": 0.0}
            )
        for entry in info.get("rejections", []):
            self._trade_history.append(
                {
                    "step": self._step_count,
                    "kind": "rejection",
                    **entry,
                    "reward": 0.0,
                }
            )
        for entry in info.get("expired", []):
            self._trade_history.append(
                {
                    "step": self._step_count,
                    "kind": "expiration",
                    **entry,
                    "reward": 0.0,
                }
            )
        self._done = bool(info.get("done", False))
        # Episode-end auto-expire: GTC orders still queued are cancelled.
        # Transactional offline routes orders through the executor; legacy
        # {sym:int} never queues, so this is a no-op there.
        if self._done:
            executor = getattr(self, "_executor", None)
            if executor is not None:
                prices = (
                    self._current_prices() if hasattr(self, "_current_prices") else {}
                )
                extra_expired = executor.expire_all(prices, step=self._step_count)
                if extra_expired:
                    self._sync_portfolio_from_executor()
                    expired_dicts = [r.to_dict() for r in extra_expired]
                    info.setdefault("expired", []).extend(expired_dicts)
                    for entry in expired_dicts:
                        self._trade_history.append(
                            {
                                "step": self._step_count,
                                "kind": "episode_end_expiration",
                                **entry,
                                "reward": 0.0,
                            }
                        )
        obs = self._get_market_observation() if not self._done else None
        return obs, float(reward), self._done, info

    #: Reward-bank :class:`TradingReward` classes dispatched by
    #: :meth:`_compute_trading_metrics`. Each produces one output key
    #: named after the metric's ``.name`` attribute (``pnl``,
    #: ``sharpe_ratio``, ``max_drawdown``, ``win_rate`` for the default
    #: set). Subclasses extend with
    #: ``(*TradingTask.DEFAULT_TRADING_REWARDS, MyReward)``.
    DEFAULT_TRADING_REWARDS: tuple[type[TradingReward], ...] = (
        PnL,
        SharpeRatio,
        MaxDrawdown,
        WinRate,
    )

    def evaluate(self, agent_actions: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Compute trading metrics from accumulated trade history.

        Does NOT replay the episode -- reads from ``self._trade_history``
        which is populated during :meth:`step`. Scalar stats
        (``total_return``, ``num_trades``, ``mean_return``, ``std_return``)
        are computed inline; distributional metrics go through the reward
        bank via :attr:`DEFAULT_TRADING_REWARDS` (pure-Python, no torch).
        ``total_pnl`` is the full PnL across *all* symbols (diagnostic);
        the reward-bank metrics above are scoped to ``target_symbols``.
        """
        metrics = self._compute_trading_metrics(self._trade_history)
        metrics["total_pnl"] = float(
            self._executor.compute_pnl(self._current_prices()).get("__total", 0.0)
        )
        return metrics

    def _compute_trading_metrics(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Trading metrics from a trade-history stream.

        When ``target_symbols`` is a strict subset of ``symbols``, each
        step's ``reward`` is replaced by the sum of its per-symbol
        contributions across the target subset (the full history is kept
        as audit log); otherwise the aggregate stream is scored directly.
        """
        target = getattr(self, "_target_symbols", None)
        symbols = getattr(self, "_symbols", None)
        if not target or not symbols or set(target) == set(symbols):
            return _compute_trading_metrics_from_history(
                history,
                self.DEFAULT_TRADING_REWARDS,
                count_label="num_trades",
            )
        target_set = set(target)
        filtered: List[Dict[str, Any]] = []
        for entry in history:
            per_sym = entry.get("per_symbol_rewards") or {}
            tgt_reward = float(sum(v for s, v in per_sym.items() if s in target_set))
            filtered.append({**entry, "reward": tgt_reward})
        return _compute_trading_metrics_from_history(
            filtered,
            self.DEFAULT_TRADING_REWARDS,
            count_label="num_trades",
        )


class BaseEvaluator(ABC):
    REWARD_NAMES: list[str] = []

    @abstractmethod
    def score(
        self,
        predictions: Any,
        ground_truth: Any,
        weights: list[float] | None = None,
        reward_output: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        raise NotImplementedError


class TaskEnvironmentAdapter:
    def __init__(self, task: BaseTask) -> None:
        self._task = task
        self._episode_actions: list[Any] = []

    def reset(self) -> Any:
        self._episode_actions = []
        return self._task.reset()

    def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:
        self._episode_actions.append(action)
        return self._task.step(action)

    def get_rewards(self) -> Dict[str, float]:
        return self._task.evaluate(self._episode_actions)

    @property
    def episode_actions(self) -> list[Any]:
        return list(self._episode_actions)


class BaseAgent(ABC):
    """Interface for agents that interact with tasks via the gym API.

    3rd-party agents implement this minimal protocol.  The only required
    method is :meth:`act`; the lifecycle hooks are optional.
    """

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Given an observation, return an action."""
        raise NotImplementedError

    def on_episode_start(self, task_metadata: TaskMetadata) -> None:
        """Called before ``task.reset()``.  Override to inspect task info."""

    def on_episode_end(self, rewards: Dict[str, float]) -> None:
        """Called after ``task.evaluate()``.  Override to observe final rewards."""
