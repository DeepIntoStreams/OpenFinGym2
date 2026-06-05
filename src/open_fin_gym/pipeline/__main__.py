import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="pipeline_config"
)
def run_pipeline(cfg: DictConfig) -> None:
    """Main pipeline hydra entrypoint"""
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    run_pipeline()
