# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

Module for preparing datasets and dataloaders in PyTorch.

This script includes:
- Custom dataset definition (`MyDataset`)
- Transform management (`MyTransforms`)
- Functions to create datasets and data loaders

@author: tadahaya
"""
import numpy as np
from typing import Tuple, Optional, List, Union

import torch
import torchvision.transforms as transforms

# frozen
class MyDataset(torch.utils.data.Dataset):
    """
    Custom dataset implementation for supervised and unsupervised tasks.

    Args:
        data (np.ndarray): Array containing the data samples.
        label (Optional[np.ndarray]): Array containing labels for supervised learning.
                                      Defaults to `None` for unsupervised learning.
        transform (Optional[Union[transforms.Compose, List[callable]]]): 
                 A single transformation or a list of transformations applied to each sample.
    """
    def __init__(self, data: np.ndarray, label: Optional[np.ndarray] = None, transform=None):
        if data is None:
            raise ValueError("`data` cannot be None. Please provide the input data.")
        if label is None:
            label = np.full(len(data), np.nan)  # Assign NaN for unsupervised learning
        if not isinstance(transform, list):
            self.transform = [transform]
        else:
            self.transform = transform
        self.data = data
        self.label = label
        self.datanum = len(self.data)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.datanum

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Retrieves a single data sample and its corresponding label.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            Tuple[torch.Tensor, float]: Transformed data sample and its label.
        """
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            for t in self.transform:
                if t is not None:
                    out_data = t(out_data)
        return out_data, out_label


class MyTransforms:
    """
    Example transformation class that converts numpy arrays to PyTorch tensors.

    This can be extended with additional transformations as needed.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a PyTorch tensor.
        
        Args:
            x (np.ndarray): Input numpy array.
        
        Returns:
            torch.Tensor: Converted tensor.
        """
        return torch.from_numpy(x.astype(np.float32))


def prep_dataset(data: np.ndarray, label: Optional[np.ndarray] = None, transform=None) -> MyDataset:
    """
    Prepares a PyTorch dataset from raw data and labels.

    Args:
        data (np.ndarray): Input data samples.
        label (Optional[np.ndarray]): Labels for supervised learning, defaults to None.
        transform (Optional[Union[transforms.Compose, List[callable]]]): 
                  Transformations to apply to the data samples.

    Returns:
        MyDataset: A dataset instance.
    """
    return MyDataset(data, label, transform)


def prep_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Prepares a PyTorch DataLoader for training or testing.

    Args:
        dataset (torch.utils.data.Dataset): Dataset instance.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads for data loading.
        pin_memory (bool): Whether to use pinned memory for faster data transfer.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader instance.
    """
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )


def _worker_init_fn(worker_id: int):
    """
    Initializes each worker with a unique random seed to ensure reproducibility.

    Args:
        worker_id (int): Unique identifier for the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_data(
    train_x: np.ndarray,
    train_y: Optional[np.ndarray],
    test_x: np.ndarray,
    test_y: Optional[np.ndarray],
    batch_size: int,
    transform: Tuple[Optional[List[callable]], Optional[List[callable]]] = (None, None),
    shuffle: Tuple[bool, bool] = (True, False),
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares training and testing data loaders from raw data.

    Args:
        train_x (np.ndarray): Training data samples.
        train_y (Optional[np.ndarray]): Training labels (or None for unsupervised learning).
        test_x (np.ndarray): Testing data samples.
        test_y (Optional[np.ndarray]): Testing labels (or None for unsupervised learning).
        batch_size (int): Number of samples per batch.
        transform (Tuple[Optional[List[callable]], Optional[List[callable]]]): 
                  Transformations for training and testing data, respectively.
        shuffle (Tuple[bool, bool]): Whether to shuffle training and testing data.
        num_workers (int): Number of worker threads for data loading.
        pin_memory (bool): Whether to use pinned memory for faster data transfer.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
        Data loaders for training and testing.
    """
    train_dataset = prep_dataset(train_x, train_y, transform[0])
    test_dataset = prep_dataset(test_x, test_y, transform[1])
    train_loader = prep_dataloader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle[0], 
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = prep_dataloader(
        dataset=test_dataset, batch_size=batch_size, shuffle=shuffle[1], 
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader