"""Epoch-reconstructible batch orders used by the reference notebook."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Sampler


class NotebookBatchSampler(Sampler):
    def __init__(self, records: int, batch_size: int, seed: int, protocol: str):
        self.records, self.batch_size, self.seed = records, batch_size, seed
        self.protocol, self.epoch = protocol, 0
        if protocol not in {"ssl", "finetune", "linear"}:
            raise ValueError("Unknown notebook sampling protocol")

    def __len__(self):
        return (
            self.records // self.batch_size
            if self.protocol == "ssl"
            else math.ceil(self.records / self.batch_size)
        )

    def __iter__(self):
        if self.protocol == "ssl":
            order = np.random.RandomState(self.seed * 1000 + self.epoch).permutation(self.records)
        elif self.protocol == "finetune":
            rng = np.random.RandomState(self.seed)
            for _ in range(self.epoch + 1):
                order = rng.permutation(self.records)
        else:
            generator = torch.Generator().manual_seed(self.seed)
            for _ in range(self.epoch + 1):
                order = torch.randperm(self.records, generator=generator).numpy()
        for i in range(len(self)):
            batch = order[i * self.batch_size : (i + 1) * self.batch_size]
            yield (batch if self.protocol == "linear" else np.sort(batch)).tolist()
