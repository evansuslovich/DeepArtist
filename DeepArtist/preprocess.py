#!/usr/bin/env python3

import os
from collections import OrderedDict
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


    def length(self) -> int:
        return len(self.labels())


    def label_map(self) -> list[str]:

        def extract_label(label_tuple: str) -> str:
            return os.path.basename(os.path.dirname(label_tuple[0]))
        
        ungrouped_labels = map(extract_label, self.imgs)

        label_map = []

        for label in ungrouped_labels:
            if label not in label_map:
                label_map.append(label)
        
        return label_map




transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((224, 224), antialias=True),  # Explicitly set antialias to True
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust if images are grayscale
])


def split_dataset(dataset, train_proportion=0.8, validate_size=0.1):
    indices = list(range(len(dataset)))
    train_idx, posttrain_idx = train_test_split(indices, test_size=train_proportion)
    posttrain_dataset = Subset(dataset, posttrain_idx)

    indices = list(range(len(posttrain_dataset)))

    validate_idx, test_idx = train_test_split(indices, test_size=validate_size / (1 - train_proportion))

    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['validation'] = Subset(posttrain_dataset, validate_idx)
    datasets['test'] = Subset(posttrain_dataset, test_idx)

    return datasets


def load(root_dir: str, batch_size: int = 100):

    dataset = LabeledImageDataset(root_dir=root_dir, transform=transform)

    label_map = dataset.label_map()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    datatests = split_dataset(dataset)

    print(datatests['train'].targets())

    return loader

if __name__ == '__main__':

    loader = load(ROOT)
