from datetime import datetime
from pathlib import Path

import hydra
from sqlalchemy import create_engine

from .config import PipelineConfig
from .db.tables import Base
from .steps.retrieval.pipeline import PaperRetrieval
from .steps.scrape_papers.pipeline import PaperScrapingPipeline


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="pipeline_config"
)
def run_pipeline(cfg: PipelineConfig) -> None:
    """
    Main pipeline hydra entrypoint for the task generation pipeline

    Args:
        cfg: Configuration
    """
    db_engine = create_engine(cfg.db_engine, pool_size=150)
    Base.metadata.create_all(db_engine)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    scraping_pipeline = PaperScrapingPipeline(db_engine, cfg.scraping)
    scraping_pipeline.run(
        output_dir,
        cfg.scraping.scopes,
        datetime.strptime(cfg.scraping.since, "%Y-%m-%d"),
        datetime.strptime(cfg.scraping.until, "%Y-%m-%d"),
        cfg.scraping.max_papers_per_scope,
    )

    retrieval_pipeline = PaperRetrieval(db_engine)
    retrieval_pipeline.download_and_chunk_papers(output_dir)


if __name__ == "__main__":
    run_pipeline()
