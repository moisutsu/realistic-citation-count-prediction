import math
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from src.data.datasets import JsonlDataset
from src.utils import Discretizer
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast


class CitationCountRegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_dir: str,
        tokenizer: BertTokenizerFast,
        max_length: int,
        num_workers: int,
        use_citation_score: bool,
        use_published_date: bool,
        use_author_score: bool,
        author_score_label_count: Optional[int],
        use_trend_score: bool,
        trend_score_column_name: str,
        trend_score_labels: Optional[list[int]],
        use_elapsed_months_for_trend_score: Optional[list[int]],
    ):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers
        self.use_citation_score = use_citation_score
        self.use_published_date = use_published_date
        self.use_author_score = use_author_score
        self.author_score_label_count = author_score_label_count
        self.use_trend_score = use_trend_score
        self.trend_score_column_name = trend_score_column_name
        self.trend_score_labels = trend_score_labels
        self.use_elapsed_months_for_trend_score = use_elapsed_months_for_trend_score

        self.train_dataset: Dataset
        self.valid_dataset: Dataset
        self.test_dataset: Dataset

    def transform_train(self, lines: list[dict]):
        dataset = []

        if self.use_author_score:
            if self.author_score_label_count is None:
                raise ValueError(
                    "When use_author_score is specified, author_score_label_count must be specified"
                )

            self.discretizer = Discretizer()
            author_scores = [float(line["author_score"]) for line in lines]
            self.discretizer.set_boundary_value(
                author_scores, self.author_score_label_count
            )

            additional_special_tokens = [
                f"[AUTHOR_SCORE_{label}]" for label in self.discretizer.labels
            ]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens}
            )

        if self.use_trend_score:
            additional_special_tokens = [
                f"[TREND_SCORE_{label}]" for label in self.trend_score_labels
            ]
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": additional_special_tokens}
            )

        for line in lines:
            title, abstract, citation_count = (
                line["title"],
                line["abstract"],
                np.float32(line["citation_counts"]),
            )

            if self.use_author_score:
                author_score = line["author_score"]
                title = (
                    f"[AUTHOR_SCORE_{self.discretizer.number_to_label(author_score)}]"
                    + title
                )

            if self.use_trend_score:
                trend_scores = line[self.trend_score_column_name]
                title = (
                    "".join(
                        [
                            f"[TREND_SCORE_{trend_scores[elapsed_month - 1]}]"
                            for elapsed_month in self.use_elapsed_months_for_trend_score
                        ]
                    )
                    + title
                )

            ids = self.tokenizer(
                title,
                abstract,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            if self.use_citation_score:
                citation_count = np.float32(math.log(citation_count + 1))

            if self.use_published_date:
                published_date = np.float32(line["elapsed_months"])
                data = (
                    torch.tensor(ids.input_ids),
                    torch.tensor(ids.attention_mask),
                    torch.tensor(ids.get("token_type_ids", create_token_type_ids_from_sequences(self.tokenizer, ids.input_ids))),
                    citation_count,
                    published_date,
                )
            else:
                data = (
                    torch.tensor(ids.input_ids),
                    torch.tensor(ids.attention_mask),
                    torch.tensor(ids.get("token_type_ids", create_token_type_ids_from_sequences(self.tokenizer, ids.input_ids))),
                    citation_count,
                )

            dataset.append(data)

        return dataset

    def transform_eval(self, lines: list[dict]):
        dataset = []
        for line in lines:
            title, abstract, citation_count = (
                line["title"],
                line["abstract"],
                np.float32(line["citation_counts"]),
            )

            if self.use_author_score:
                author_score = line["author_score"]
                title = (
                    f"[AUTHOR_SCORE_{self.discretizer.number_to_label(author_score)}]"
                    + title
                )

            if self.use_trend_score:
                trend_scores = line[self.trend_score_column_name]
                title = (
                    "".join(
                        [
                            f"[TREND_SCORE_{trend_scores[elapsed_month - 1]}]"
                            for elapsed_month in self.use_elapsed_months_for_trend_score
                        ]
                    )
                    + title
                )

            ids = self.tokenizer(
                title,
                abstract,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            if self.use_published_date:
                published_date = np.float32(line["elapsed_months"])
                data = (
                    torch.tensor(ids.input_ids),
                    torch.tensor(ids.attention_mask),
                    torch.tensor(ids.get("token_type_ids", )),
                    published_date,
                    citation_count,
                )
            else:
                data = (
                    torch.tensor(ids.input_ids),
                    torch.tensor(ids.attention_mask),
                    torch.tensor(ids.get("token_type_ids", create_token_type_ids_from_sequences(self.tokenizer, ids.input_ids))),
                    citation_count,
                )

            dataset.append(data)

        return dataset

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit":
            self.train_dataset = JsonlDataset(
                self.dataset_dir / "train.jsonl", self.transform_train
            )
            self.valid_dataset = JsonlDataset(
                self.dataset_dir / "valid.jsonl", self.transform_eval
            )

    def setup_test(self, test_name: str):
        self.test_name = test_name
        self.test_dataset = JsonlDataset(
            self.dataset_dir / f"{test_name}.jsonl", self.transform_eval
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

def create_token_type_ids_from_sequences(tokenizer: BertTokenizerFast, sequence: list[int]) -> list[int]:
    return tokenizer.create_token_type_ids_from_sequences(sequence)[:len(sequence)]
