from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
from dotenv import load_dotenv
from sqlalchemy import create_engine

from .config import PipelineConfig
from .db.tables import Base
from .steps.judge.pipeline import Judge
from .steps.retrieval.pipeline import PaperRetrieval
from .steps.scrape_papers.pipeline import PaperScrapingPipeline
from .steps.task_extraction.pipeline import TaskExtractor
from .steps.task_generator.pipeline import TaskGenerator


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="pipeline_config"
)
def run_pipeline(cfg: PipelineConfig) -> None:
    """
    Main pipeline hydra entrypoint for the task generation pipeline

    Args:
        cfg: Configuration
    """

    use_mlflow = "mlflow" in cfg

    if use_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.langchain.autolog()
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    load_dotenv()

    db_engine = create_engine(cfg.db_engine)
    Base.metadata.create_all(db_engine)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    scopes = [scope for scope in cfg.scopes if scope.enabled]

    context = mlflow.start_run() if use_mlflow else nullcontext()

    with context:
        scraping_pipeline = PaperScrapingPipeline(db_engine, cfg.scraping)
        scraping_pipeline.run(
            output_dir,
            scopes,
            datetime.strptime(cfg.scraping.since, "%Y-%m-%d"),
            datetime.strptime(cfg.scraping.until, "%Y-%m-%d"),
            cfg.scraping.max_papers_per_scope,
        )

        retrieval_pipeline = PaperRetrieval(db_engine)
        retrieval_pipeline.download_and_chunk_papers(output_dir)

        judge_pipeline = Judge(db_engine, cfg.judge)
        judge_pipeline.run(output_dir, scopes)

        task_extractor = TaskExtractor(db_engine, cfg.task_extractor)
        task_extractor.run(output_dir, scopes)

        task_generator = TaskGenerator(db_engine, cfg.task_generator)
        task_generator.run(output_dir, scopes)


if __name__ == "__main__":
    run_pipeline()
