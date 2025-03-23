import gc
import math
import os
import statistics
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from scipy import stats
from sklearn.metrics import mean_squared_error
from torch import nn
from transformers import BertTokenizerFast, ProcessorMixin

from .metrics import top_k_percent_acc


class Experiment(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        # Set logger
        self.mlflow_logger = self.create_logger()
        self.trainer: pl.Trainer = instantiate(
            self.config.trainer,
            logger=self.mlflow_logger,
        )

        self.log_all_hyperparams()
        self.log_all_artifact()

    def initialize_experiment_settings(self):
        seed_everything(self.config.seed + self.running_id)

        callbacks = []
        self.early_stop_callback: EarlyStopping = instantiate(
            self.config.early_stop_callback,
            monitor=f"{self.config.monitor}{self.log_suffix}",
        )
        if self.config.enable_early_stopping:
            callbacks.append(self.early_stop_callback)

        self.checkpoint_callback: ModelCheckpoint = instantiate(
            self.config.checkpoint_callback,
            dirpath=f"checkpoints{self.log_suffix}",
            monitor=f"{self.config.monitor}{self.log_suffix}",
        )
        callbacks.append(self.checkpoint_callback)

        self.trainer: pl.Trainer = instantiate(
            self.config.trainer,
            callbacks=callbacks,
            logger=self.mlflow_logger,
        )

        self.model: nn.Module = instantiate(self.config.model)
        self.tokenizer: BertTokenizerFast = instantiate(self.config.tokenizer)
        self.datamodule: pl.LightningDataModule = instantiate(
            self.config.datamodule, tokenizer=self.tokenizer, num_workers=os.cpu_count()
        )

    def run(self):
        results = []
        for running_id in range(self.config.experiment_count):
            self.running_id = running_id
            self.log_suffix = (
                "" if self.config.experiment_count == 1 else f"_exp-{self.running_id}"
            )

            self.initialize_experiment_settings()

            self.fit()
            result = self.test()
            results.append(result)

            self.release_gpu_memory()

        self.log_results_of_multiple_experiments(results)

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = instantiate(
            self.config.optimizer, params=self.model.parameters()
        )
        lr_scheduler = instantiate(
            self.config.lr_scheduler,
            optimizer=optimizer,
            dataset_length=len(self.datamodule.train_dataset),
            batch_size=self.config.batch_size
            * self.config.trainer.accumulate_grad_batches,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, _):
        loss = self.model.fit(*batch)
        self.logger.log_metrics(
            {f"training_loss{self.log_suffix}": loss.item()}, self.global_step
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, _):
        *inputs, citation_counts = batch
        score = self.model(*inputs)

        return score.flatten().tolist(), citation_counts.tolist()

    def validation_epoch_end(self, outputs):
        scores, citation_counts = [], []
        for score, citation_count in outputs:
            scores.extend(score)
            citation_counts.extend(citation_count)

        rank_correlation, _ = stats.spearmanr(scores, citation_counts)
        top_10_percent_acc = top_k_percent_acc(10, citation_counts, scores)
        mse = mean_squared_error(
            scores, [math.log(citation_count + 1) for citation_count in citation_counts]
        )

        self.logger.log_metrics(
            {
                f"valid_rank_correlation{self.log_suffix}": rank_correlation,
                f"valid_top_10_percent_acc{self.log_suffix}": top_10_percent_acc,
                f"valid_mse{self.log_suffix}": mse,
            },
            self.current_epoch,
        )

        self.log(f"{self.config.monitor}{self.log_suffix}", rank_correlation)

        return rank_correlation

    @torch.no_grad()
    def test_step(self, batch, _):
        *inputs, citation_counts = batch
        score = self.model(*inputs)

        return score.flatten().tolist(), citation_counts.tolist()

    def test_epoch_end(self, outputs):
        scores, citation_counts = [], []
        for score, citation_count in outputs:
            scores.extend(score)
            citation_counts.extend(citation_count)

        predict_results_path = f"predict_results-exp_{self.running_id}.txt"
        Path(predict_results_path).write_text(
            "".join([f"{score}\n" for score in scores])
        )
        self.mlflow_log_artifact(predict_results_path)

        rank_correlation, _ = stats.spearmanr(scores, citation_counts)
        top_10_percent_acc = top_k_percent_acc(10, citation_counts, scores)
        mse = mean_squared_error(
            scores, [math.log(citation_count + 1) for citation_count in citation_counts]
        )

        self.logger.log_metrics(
            {
                f"{self.datamodule.test_name}_rank_correlation{self.log_suffix}": rank_correlation,
                f"{self.datamodule.test_name}_top_10_percent_acc{self.log_suffix}": top_10_percent_acc,
                f"{self.datamodule.test_name}_mse{self.log_suffix}": mse,
            },
            self.current_epoch,
        )

        self.log(
            f"inner_{self.datamodule.test_name}_rank_correlation",
            rank_correlation,
        )
        self.log(
            f"inner_{self.datamodule.test_name}_top_10_percent_acc",
            top_10_percent_acc,
        )
        self.log(
            f"inner_{self.datamodule.test_name}_mse",
            mse,
        )

        return {
            f"{self.datamodule.test_name}_rank_correlation": rank_correlation,
            f"{self.datamodule.test_name}_top_10_percent_acc": top_10_percent_acc,
            f"{self.datamodule.test_name}_mse": mse,
        }

    def fit(self):
        self.datamodule.setup("fit")

        # For when you add a new special token
        if self.config.resize_token_embeddings:
            if isinstance(self.tokenizer, ProcessorMixin):
                self.model.resize_token_embeddings(len(self.tokenizer.tokenizer))
            else:
                self.model.resize_token_embeddings(len(self.tokenizer))

        # For when extending position embeddings to increase input length
        if self.config.resize_position_embeddings:
            self.model.resize_position_embeddings(self.config.datamodule.max_length + 2)
            self.tokenizer.tokenizer.model_max_length = (
                self.config.datamodule.max_length
            )
            self.model.model.config.max_position_embeddings = (
                self.config.datamodule.max_length
            )

        self.trainer.fit(self, datamodule=self.datamodule)

        if self.config.save_model:
            save_path = f"model{self.log_suffix}.torch"
            torch.save(self.model.state_dict(), save_path)
            self.mlflow_log_artifact(save_path)

    def test(self) -> dict[str, float]:
        results = dict()
        for test_name in self.config.test_names:
            self.datamodule.setup_test(test_name)
            result = self.trainer.test(self, datamodule=self.datamodule)
            results |= result[0]
        return results

    def mlflow_log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)

    def log_all_hyperparams(self):
        hyperparams = {}

        global_keys = ["bert_model", "experiment_count", "dataset_name", "seed"]
        local_keys = {
            "model": [
                "pooling_method",
                "pooling_method_for_multiple_output",
                "pooling_method_for_single_output",
                "classifier_dropout",
            ],
            "datamodule": ["mode", "max_length", "max_page"],
            "optimizer": ["lr"],
            "lr_scheduler": ["linear_warmup"],
        }

        for global_key in global_keys:
            if global_key in self.config:
                hyperparams[global_key] = getattr(self.config, global_key)

        for local_key, local_values in local_keys.items():
            if not local_key in self.config:
                continue

            local_config = getattr(self.config, local_key)

            for local_value in local_values:
                if local_value in local_config:
                    hyperparams[local_value] = getattr(local_config, local_value)

        hyperparams["batch_size"] = (
            self.config.batch_size * self.config.trainer.accumulate_grad_batches
        )

        self.logger.log_hyperparams(hyperparams)

    def log_all_artifact(self):
        # Hydra
        self.mlflow_log_artifact(".hydra/config.yaml")
        self.mlflow_log_artifact(".hydra/hydra.yaml")
        self.mlflow_log_artifact(".hydra/overrides.yaml")

    def log_results_of_multiple_experiments(self, results: list[dict[str, float]]):
        if self.config.experiment_count == 1:
            return

        for inner_test_name in results[0].keys():
            test_name = inner_test_name.removeprefix("inner_")

            test_values = [result[inner_test_name] for result in results]

            self.logger.log_metrics(
                {
                    f"{test_name}_mean": statistics.mean(test_values),
                    f"{test_name}_stdev": statistics.stdev(test_values),
                }
            )

    def create_logger(self) -> LightningLoggerBase:
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/5367#issuecomment-757990645
        user_specified_tags = {}
        user_specified_tags[MLFLOW_RUN_NAME] = self.config.run_name
        tags = context_registry.resolve_tags(user_specified_tags)

        logger = instantiate(self.config.logger, tags=tags)
        return logger

    @staticmethod
    def release_gpu_memory():
        gc.collect()
        torch.cuda.empty_cache()
