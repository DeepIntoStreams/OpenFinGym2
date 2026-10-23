import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from sqlalchemy import create_engine

from open_fin_gym.pipeline.db.tables import Base, PaperStatus
from open_fin_gym.pipeline.db.utils import set_paper_status
from open_fin_gym.pipeline.steps.judge.pipeline import Judge
from open_fin_gym.pipeline.steps.retrieval.pipeline import PaperRetrieval
from open_fin_gym.pipeline.steps.scrape_papers.pipeline import (
    PaperScrapingPipeline,
)
from open_fin_gym.pipeline.steps.task_critic.pipeline import TaskCritic
from open_fin_gym.pipeline.steps.task_export.pipeline import TaskExporter
from open_fin_gym.pipeline.steps.task_extraction.pipeline import TaskExtractor
from open_fin_gym.pipeline.steps.task_generator.pipeline import TaskGenerator

from .config import ManualPipelineConfig, Scope
from .utils import check_docker_available, init_context

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="manual_pipeline_config"
)
def run_manual_pipeline(cfg: ManualPipelineConfig) -> None:
    """
    Main pipeline hydra entrypoint for the task generation pipeline

    Args:
        cfg: Configuration
    """
    check_docker_available(logger)  # Check docker server is available
    load_dotenv()
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    context = init_context(cfg, logger)
    assert len(cfg.scopes) == 1, "A single scope should be selected"
    scope = Scope(**[*cfg.scopes.values()][0])
    assert cfg.arxiv_id is not None, "Field `arxiv_id` should be set"
    db_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(db_engine)

    with context:
        logger.info(f"Starting pipeline for paper {cfg.arxiv_id}")

        scraping_pipeline = PaperScrapingPipeline(db_engine, cfg.scraping)
        paper = scraping_pipeline.scrape_paper(output_dir, scope, cfg.arxiv_id)

        retrieval_pipeline = PaperRetrieval(db_engine, cfg.retrieval)
        retrieval_pipeline.download_and_chunk_papers(output_dir)

        if cfg.enable_judge:
            judge_pipeline = Judge(db_engine, cfg.judge)
            judge_pipeline.run(output_dir, [scope])
        else:
            logger.info("Skipping paper judgement stage and auto accepting paper")
            set_paper_status(db_engine, paper, PaperStatus.ACCEPTED)

        task_extractor = TaskExtractor(db_engine, cfg.task_extractor)
        task_extractor.run(output_dir, [scope])

        task_critic = TaskCritic(db_engine, cfg.task_critic)
        if cfg.enable_task_critic:
            task_critic.run(output_dir, [scope])
        else:
            task_critic.run(output_dir, [scope], auto_approve=True)

        task_generator = TaskGenerator(db_engine, cfg.task_generator)
        task_generator.run(output_dir, [scope])

        task_exporter = TaskExporter(db_engine, cfg.task_exporter)
        task_exporter.run(output_dir, [scope])


if __name__ == "__main__":
    run_manual_pipeline()
