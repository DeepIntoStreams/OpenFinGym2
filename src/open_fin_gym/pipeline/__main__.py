import logging
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from sqlalchemy import create_engine

from .config import PipelineConfig, Scope
from .db.tables import Base
from .steps.judge.pipeline import Judge
from .steps.retrieval.pipeline import PaperRetrieval
from .steps.scrape_papers.pipeline import PaperScrapingPipeline
from .steps.task_critic.pipeline import TaskCritic
from .steps.task_export.pipeline import TaskExporter
from .steps.task_extraction.pipeline import TaskExtractor
from .steps.task_generator.pipeline import TaskGenerator
from .utils import check_docker_available, init_context

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="pipeline_config"
)
def run_pipeline(cfg: PipelineConfig) -> None:
    """
    Main pipeline hydra entrypoint for the task generation pipeline

    Args:
        cfg: Configuration
    """
    load_dotenv()

    # Check docker server is available
    check_docker_available(logger)

    db_engine = create_engine(cfg.db_engine)
    Base.metadata.create_all(db_engine)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    scopes = [Scope(**scope) for scope in cfg.scopes.values() if scope.enabled]
    context = init_context(cfg, logger)

    with context:
        logger.info("Starting pipeline")
        scraping_pipeline = PaperScrapingPipeline(db_engine, cfg.scraping)
        scraping_pipeline.run(
            output_dir,
            scopes,
            datetime.strptime(cfg.scraping.since, "%Y-%m-%d"),
            datetime.strptime(cfg.scraping.until, "%Y-%m-%d"),
            cfg.scraping.max_papers_per_scope,
        )

        retrieval_pipeline = PaperRetrieval(db_engine, cfg.retrieval)
        retrieval_pipeline.download_and_chunk_papers(output_dir)

        judge_pipeline = Judge(db_engine, cfg.judge)
        judge_pipeline.run(output_dir, scopes)

        task_extractor = TaskExtractor(db_engine, cfg.task_extractor)
        task_extractor.run(output_dir, scopes)

        task_critic = TaskCritic(db_engine, cfg.task_critic)
        if cfg.enable_task_critic:
            task_critic.run(output_dir, scopes)
        else:
            task_critic.run(output_dir, scopes, auto_approve=True)

        task_generator = TaskGenerator(db_engine, cfg.task_generator)
        task_generator.run(output_dir, scopes)

        task_exporter = TaskExporter(db_engine, cfg.task_exporter)
        task_exporter.run(output_dir, scopes)


if __name__ == "__main__":
    run_pipeline()
