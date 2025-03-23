from typing import Union

import pandas as pd


class Discretizer:
    def set_boundary_value(self, numbers: list[float], label_count: int):
        _, self.bins = pd.qcut(numbers, label_count, retbins=True, duplicates="drop")

        self.acutual_label_count = len(self.bins) - 1

        self.labels = [i for i in range(1, self.acutual_label_count + 1)]
        # To avoid creating nan
        self.bins[0] = -float("inf")
        self.bins[-1] = float("inf")

    def number_to_label(
        self, number: Union[float, list[float]]
    ) -> Union[int, list[int]]:
        if isinstance(number, list):
            return list(pd.cut(number, self.bins, labels=self.labels))
        else:
            return list(pd.cut([number], self.bins, labels=self.labels))[0]

    def __call__(self, number: Union[float, list[float]]) -> Union[int, list[int]]:
        return self.number_to_label(number)
