#!/usr/bin/env python3

import torch
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


ROOT = './DATA'

class LabeledImageDataset(ImageFolder):

    def __init__(self, root_dir: str, transform) -> None:
        super().__init__(root_dir, transform)
    
    def labels(self) -> list[str]:
        return self.targets

    def label_map(self) -> list[str]:

        def extract_label(path: str) -> str:
            return path

        return list(map(extract_label, self.imgs))


transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224), antialias=True),  # Explicitly set antialias to True
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust if images are grayscale
])


def split_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def load(root_dir: str, batch_size: int = 100):

    dataset = LabeledImageDataset(root_dir=root_dir, transform=transform)

    print(dataset.labels())
    print(dataset.label_map())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


if __name__ == '__main__':

    loader = load(ROOT)
