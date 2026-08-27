from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import hydra
import ray
from dataset import HarborTaskDataset
from harbor_generator import HarborGenerator
from omegaconf import DictConfig, open_dict
from skyrl.train.config import (
    GeneratorConfig,
    SkyRLTrainConfig,
    get_config_as_yaml_str,
)
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from skyrl.train.utils.utils import initialize_ray


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


class HarborFullyAsyncExp(HarborExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = HarborFullyAsyncExp(cfg)
    exp.run()


@hydra.main(config_path="./", config_name="default_config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    with open_dict(cfg):
        cfg.harbor_trial_config.trials_dir = str(output_dir / "trials_run/")
        cfg.trainer.export_path = str(output_dir / "exports/")
        cfg.trainer.ckpt_path = str(output_dir / "ckpts/")
        cfg.trainer.log_path = str(output_dir / "logs/")

    cfg = HarborSkyRLConfig.from_dict_config(cfg)

    with open(output_dir / "compiled_cfg.yaml", "w") as f:
        f.write(get_config_as_yaml_str(cfg))

    validate_cfg(cfg)

    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )

    initialize_ray(cfg)

    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
