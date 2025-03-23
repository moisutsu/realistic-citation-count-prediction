import hydra
import torch
from omegaconf import DictConfig
from rich.traceback import install as rich_install

from src import Experiment
from src.utils import print_config

torch.multiprocessing.set_sharing_strategy("file_system")

rich_install()


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    if config.get("print_config"):
        print_config(config, resolve=True)

    exp = Experiment(config)

    exp.run()


if __name__ == "__main__":
    main()
