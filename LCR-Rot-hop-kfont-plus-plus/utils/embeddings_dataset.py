import glob
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EmbeddingsDataset(Dataset):
    def __init__(self, year: int, domain: str = "restaurants", phase='Train', device=torch.device('cpu'),
                 empty_ok=False, enable_cache=True):
        self.dir = f'data/embeddings/{year}-{domain}-{phase}'
        self.device = device
        self.length = len(glob.glob(f'{self.dir}/*.pt'))
        self.cache: dict[int, tuple] = {}
        self.enable_cache = enable_cache

        if not empty_ok and self.length == 0:
            raise ValueError(f"Could not find embeddings at {self.dir}")

    def __getitem__(self, item: int):
        if item in self.cache:
            return self.cache[item]

        data: dict = torch.load(f"{self.dir}/{item}.pt", map_location=self.device, weights_only=True)
        label: torch.Tensor = torch.tensor(data['label'], requires_grad=False, device=self.device)
        sentence = data['sentence']
        target_index_start: int = data['target_from']
        target_index_end: int = data['target_to']
        hops = None

        result = (
            (sentence, target_index_start, target_index_end),
            label,
            hops
        )

        if self.enable_cache:
            self.cache[item] = result

        return result

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"EmbeddingsDataset({self.dir})"


def train_validation_split(dataset, test_size=0.2, random_state=42):
    # 获取所有样本的索引
    indices = list(range(len(dataset)))
    
    # 随机打乱索引
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    # 计算验证集的样本数量
    split = int(np.floor(test_size * len(dataset)))
    
    # 划分训练集和验证集
    train_idx, validation_idx = indices[split:], indices[:split]
    
    return train_idx, validation_idx
