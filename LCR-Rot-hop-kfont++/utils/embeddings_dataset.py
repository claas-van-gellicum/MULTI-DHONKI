import glob
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class EmbeddingsDataset(Dataset):
    def __init__(self, year: int, phase='Train', device=torch.device('cpu'),
                 empty_ok=False, enable_cache=True):
        self.dir = f'data/embeddings/{year}-{phase}'
        self.device = device
        self.length = len(glob.glob(f'{self.dir}/*.pt'))
        self.cache: dict[int, tuple] = {}
        self.enable_cache = enable_cache

        if not empty_ok and self.length == 0:
            raise ValueError(f"Could not find embeddings at {self.dir}")

    def __getitem__(self, item: int):
        if item in self.cache:
            return self.cache[item]

        data: dict = torch.load(f"{self.dir}/{item}.pt", map_location=self.device)
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


def train_validation_split(dataset: EmbeddingsDataset, validation_size=0.2, seed: Optional[float] = None):
    # create list of all labels
    loader = DataLoader(dataset, collate_fn=lambda batch: batch)
    labels: list[int] = [data[0][1].item() for data in loader]

    # create stratified train-validation split
    train_idx, validation_idx = train_test_split(
        range(len(dataset)), test_size=validation_size, shuffle=True, stratify=labels, random_state=seed)

    return train_idx, validation_idx
