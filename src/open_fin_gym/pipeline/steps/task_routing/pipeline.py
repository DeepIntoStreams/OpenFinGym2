import json
import logging
import shutil
from pathlib import Path

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from open_fin_gym import realtime
from open_fin_gym.pipeline.config import Scope
from open_fin_gym.pipeline.db.tables import (
    CURATED_TASK_TYPES,
    Chunk,
    Paper,
    PaperStatus,
)
from open_fin_gym.pipeline.db.utils import set_paper_status
from open_fin_gym.pipeline.steps.judge.utils import filter_chunks
from open_fin_gym.pipeline.steps.task_export.utils import slugify

from .config import TaskRoutingConfig
from .prompts import BundleConfig, build_routing_prompt, find_bundles

logger = logging.getLogger(__name__)


class TaskRouter:
    def __init__(self, db: Engine, cfg: TaskRoutingConfig) -> None:
        """
        Configure curated task bundles from accepted papers

        Args:
            db: SQLAlchemy db engine
            cfg: Task routing configuration
        """
        self.db = db
        self.llm: BaseChatModel = instantiate(cfg.llm)
        assert isinstance(self.llm, BaseChatModel)
        self.bundles_path = Path(cfg.bundles_path)
        self.export_path = Path(cfg.export_path)

    def run(self, output_path: Path, scopes: list[Scope]) -> None:
        """
        Route accepted papers in curated scopes onto their bundle

        Args:
            output_path: Hydra run output path
            scopes: List of paper scopes
        """
        output_path = output_path / "task_routing/"
        output_path.mkdir(exist_ok=True)

        for scope in scopes:
            if scope.task_type in CURATED_TASK_TYPES:
                self.route_scope(scope, output_path)

    def route_scope(self, scope: Scope, output_path: Path) -> None:
        """
        Configure one bundle per accepted paper in the scope

        Args:
            scope: Task scope
            output_path: Hydra run output path
        """
        bundles = find_bundles(self.bundles_path, scope.id)
        if not bundles:
            logger.error(f"No curated bundle declares scope {scope.id}")
            return

        with Session(self.db) as session:
            stmt = select(Paper).where(
                Paper.scope_id == scope.id, Paper.status == PaperStatus.ACCEPTED
            )
            papers: list[Paper] = session.execute(stmt).scalars().all()

        logger.info(f"Routing {len(papers)} papers onto {len(bundles)} bundle(s)")

        configs = []
        for paper in papers:
            with Session(self.db) as session:
                stmt = select(Chunk).where(Chunk.paper_id == paper.paper_id)
                chunks: list[Chunk] = session.execute(stmt).scalars().all()
            excerpt = "\n\n".join(x.text for x in filter_chunks(chunks))

            for bundle_path, descriptor in bundles:
                prompt = build_routing_prompt(scope, paper, descriptor, excerpt)
                llm = self.llm.with_structured_output(BundleConfig)

                try:
                    config: BundleConfig = llm.invoke(prompt)
                except Exception as e:
                    logger.error(f"Routing failed for paper {paper.paper_id}: {e}")
                    set_paper_status(self.db, paper, PaperStatus.TASK_EXTRACTION_FAILED)
                    continue

                configs.append({"paper_id": paper.paper_id, **config.model_dump()})
                if config.fit == "no_match":
                    continue

                self.write_bundle(bundle_path, paper, descriptor, config)
                set_paper_status(self.db, paper, PaperStatus.ROUTED_TO_CURATED)

        with open(output_path / f"{scope.id}.json", "w") as f:
            json.dump(configs, f, indent=4)

    def write_bundle(
        self, bundle_path: Path, paper: Paper, descriptor: dict, config: BundleConfig
    ) -> None:
        """
        Copy the bundle out with the paper's episode configuration

        Args:
            bundle_path: Source bundle directory
            paper: Paper the configuration came from
            descriptor: Parsed descriptor
            config: LLM-chosen field values
        """
        task_dir = self.export_path / slugify(
            f"{descriptor['task_id']} {paper.paper_id}"
        )
        if task_dir.exists():
            shutil.rmtree(task_dir)
        shutil.copytree(
            bundle_path, task_dir, ignore=shutil.ignore_patterns("descriptor.toml")
        )

        # The broker runs from this source inside the sidecar, so ship it rather
        # than relying on a copy left in the bundle directory.
        broker_dir = task_dir / "environment" / "realtime"
        if broker_dir.exists():
            shutil.rmtree(broker_dir)
        shutil.copytree(
            Path(realtime.__file__).parent,
            broker_dir,
            ignore=shutil.ignore_patterns("__pycache__"),
        )

        episode = {
            **descriptor["defaults"],
            **config.model_dump(exclude={"reasoning", "fit"}),
        }
        with open(task_dir / "environment" / "episode.json", "w") as f:
            json.dump(episode, f, indent=2)
