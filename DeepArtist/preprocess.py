#!/usr/bin/env python3

import os
from collections import OrderedDict
import torch
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


class LabeledImageDataset(ImageFolder):

    def __init__(self, root_dir: str, transform) -> None:
        super().__init__(root_dir, transform)
    

    def label_map(self) -> list[str]:
        '''
        Returns a list of the label (target) name strings originating from the directory name the
        images were stored in. This computes the list upon each call so this should be called once
        and the result should be stored.
        '''

        def extract_label(label_tuple: str) -> str:
            return os.path.basename(os.path.dirname(label_tuple[0]))
        
        ungrouped_labels = map(extract_label, self.imgs)

        label_map = []

        for label in ungrouped_labels:
            if label not in label_map:
                label_map.append(label)
        
        return label_map


def split_dataset(dataset,
                  train_split: float=0.8,
                  validate_split: float=0.1,
                  random_seed: int=None):
    '''
    Splits a given dataset into 3 subsets: train, validate, and test. The train_split and
    validate_split portions must add up to no more than 1.0. The size of the test set will be
    1.0 - train_split - validate_split.
    '''

    if train_split + validate_split > 1.0:
        raise ValueError(
            'The portion size of the train and validate splits must sum up to no more than 1.0.')

    dataset_all_indices = list(range(len(dataset)))
    train_idx, posttrain_idx = train_test_split(dataset_all_indices, 
                                                test_size=train_split,
                                                shuffle=True,
                                                random_state=random_seed)
    
    posttrain_dataset = Subset(dataset, posttrain_idx)
    posttrain_all_indices = list(range(len(posttrain_dataset)))
    validate_idx, test_idx = train_test_split(posttrain_all_indices,
                                              test_size=validate_split / (1 - train_split),
                                              shuffle=True,
                                              random_state=random_seed)

    return (
        Subset(dataset, train_idx),
        Subset(posttrain_dataset, validate_idx),
        Subset(posttrain_dataset, test_idx)
    )


DATA_ROOT = './Data'

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize((6, 6), antialias=True),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_artelligence_data(batch_size: int=100,
                           train_split: float=0.8,
                           validate_split: float=0.1,
                           random_seed: int=None):
    '''
    Loads the dataset, splits it into train, validate, and test subsets, then creats loaders with
    the given batch size. Returns a tuple containing the three subset loaders followed by a map of
    target values to label strings.
    '''
    
    dataset = LabeledImageDataset(root_dir=DATA_ROOT, transform=transform)
    label_map = dataset.label_map()

    train_dataset, validate_dataset, test_dataset = split_dataset(dataset,
                                                                  train_split,
                                                                  validate_split,
                                                                  random_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validate_loader, test_loader, label_map


if __name__ == '__main__':
    (
        train_loader,
        validate_loader,
        test_loader,
        label_map
    ) = load_artelligence_data(batch_size=100, train_split=0.8, validate_split=0.1, random_seed=2)

    for data in train_loader:
        print(data)

    for data in validate_loader:
        print(data)

    for data in test_loader:
        print(data)
