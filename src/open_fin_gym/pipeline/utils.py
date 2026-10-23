from contextlib import nullcontext
from logging import Logger
from typing import ContextManager

import docker
import mlflow
from docker.errors import APIError

from .config import ManualPipelineConfig, PipelineConfig


def check_docker_available(logger: Logger) -> None:
    client = docker.from_env()
    try:
        client.ping()
        logger.info("Docker server available")
    except APIError as e:
        logger.error(f"Docker server connection required to run pipeline: {e}")
    finally:
        client.close()


def init_context(
    cfg: PipelineConfig | ManualPipelineConfig,
    logger: Logger,
) -> ContextManager:

    if "mlflow" in cfg:
        logger.info("Starting MLFLow")
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.langchain.autolog()
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        return mlflow.start_run()
    else:
        return nullcontext()
