from dataclasses import dataclass


@dataclass
class ArxivConfig:
    page_size: int = 100
    sort_by: str = "SubmittedDate"
    request_interval_sec: float = 3.0
    num_retries: int = 3


@dataclass
class CrossrefConfig:
    enabled: bool = True
    base_url: str = "https://api.crossref.org"
    max_results: int = 10
    timeout_sec: int = 25
    request_delay_secs: int = 5
    min_title_similarity: float = 0.82
    mailto: str | None = None


@dataclass
class SemanticScholarConfig:
    enabled: bool = True
    batch_size: int = 500
    timeout_sec: int = 30
    api_url: str = "https://api.semanticscholar.org"
    retry: bool = True
    request_delay_secs: int = 1
