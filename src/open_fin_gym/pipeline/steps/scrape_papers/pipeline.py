from datetime import datetime
from pathlib import Path

from .collection.collector import PaperCollector
from .types import PaperRecord, Scope, ScrapingConfig


class PaperScrapingPipeline:
    def __init__(self, cfg: ScrapingConfig) -> None:
        """
        Pipeline for scraping and enriching paper-records

        Args:
            cfg: Scraping configuration
        """
        self.collector = PaperCollector(
            cfg.arxiv,
            cfg.crossref,
            cfg.semantic_scholar,
        )

    def run(
        self,
        output_path: Path,
        scopes: list[Scope],
        since_date: datetime,
        until_date: datetime,
        per_scope_limit: int,
    ) -> None:
        """
        Run the scraping process

        Args:
            output_path: Hydra run output path
            scopes: List of search scopes
            since_date: Paper search from datetime
            until_date: Paper search to datetime
            per_scope_limit: Max number of papers to return per scope
        """
        scope_papers: dict[str, tuple[Scope, dict[str, PaperRecord]]] = dict()

        for scope in scopes:
            scope_papers[scope.name] = (
                scope,
                self.collector.collect_scope(
                    scope,
                    since_date,
                    until_date,
                    per_scope_limit,
                ),
            )

        self.collector.dump_results(output_path, scope_papers)
