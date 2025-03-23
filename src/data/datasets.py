import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding


class FullTextDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        full_text_dir: Path,
        transform: Callable = lambda x: x,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.full_text_dir = full_text_dir
        self.transform = transform

        self.dataset = []

        raw_dataset = [
            json.loads(line) for line in dataset_path.read_text().splitlines()
        ]
        for line in tqdm(raw_dataset, desc="Processing dataset", dynamic_ncols=True):
            full_text_path = (
                self.full_text_dir / f"{line['publisher'].replace('/', '_')}.json"
            )

            if not full_text_path.exists():
                continue

            self.dataset.append(
                self.transform(line, json.loads(full_text_path.read_text()))
            )

        print(
            f"Number of instances where parsed full text does not exist: {len(raw_dataset) - len(self.dataset)}",
            file=sys.stderr,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


class PdfDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        encoded_pdf_dir: Path,
        transform: Callable = lambda x: x,
    ):
        super().__init__()

        self.encoded_pdf_dir = encoded_pdf_dir
        self.transform = transform

        self.dataset = []

        raw_dataset = [
            json.loads(line) for line in dataset_path.read_text().strip().split("\n")
        ]
        for line in tqdm(raw_dataset, desc="Processing dataset", dynamic_ncols=True):
            if not self.line_to_encoded_pdf_path(line).exists():
                continue

            self.dataset.append(
                self.transform(
                    line, self.load_encoded_pdf(self.line_to_encoded_pdf_path(line))
                )
            )

        print(
            f"Number of instances where encoded PDF does not exist: {len(raw_dataset) - len(self.dataset)}",
            file=sys.stderr,
        )

    def line_to_encoded_pdf_path(self, line: dict) -> Path:
        return self.encoded_pdf_dir / f"{line['publisher']}.torch"

    @staticmethod
    def load_encoded_pdf(encoded_pdf_path: Path) -> BatchEncoding:
        return torch.load(encoded_pdf_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


class TsvDataset(Dataset):
    def __init__(self, dataset_path: Path, transform: Callable = lambda x: x):
        super().__init__()
        self.dataset = []

        self.dataset.extend(
            transform(
                [
                    line.split("\t")
                    for line in dataset_path.read_text().strip().split("\n")
                ]
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


class JsonlDataset(Dataset):
    def __init__(self, dataset_path: Path, transform: Callable = lambda x: x):
        super().__init__()
        self.dataset = []

        self.dataset.extend(
            transform(
                [
                    json.loads(line)
                    for line in dataset_path.read_text().strip().split("\n")
                ]
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]
