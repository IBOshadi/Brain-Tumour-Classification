import json
from pathlib import Path

import cv2
import torch
from scipy import io as mat_io
from skimage import io
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .auto_augment import ImageNetAutoAugment
import numpy as np

from base import BaseDataLoader


# class CarsDataset(Dataset):
#     """
#         Cars Dataset
#     """
#     def __init__(self, mode, data_dir, metas, resize_width, resize_height, limit):

#         self.data_dir = data_dir
#         self.data = []
#         self.target = []

#         self.to_tensor = transforms.ToTensor()
#         self.mode = mode
#         self.resize_width = resize_width
#         self.resize_height = resize_height

#         if not isinstance(metas, str):
#             raise Exception("Train metas must be string location !")
#         labels_meta = mat_io.loadmat(metas)

#         for idx, img_ in enumerate(labels_meta['annotations'][0]):
#             if limit:
#                 if idx > limit:
#                     break

#             # self.data.append(img_resized)
#             self.data.append(data_dir + img_[5][0])
#             # if self.mode == 'train':
#             self.target.append(img_[4][0][0])

#         if self.mode == "train":
#             print("Load-up dataset with Auto Augment")
#             self.train_transform = transforms.Compose([
#                 ImageNetAutoAugment(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             ])
#         self.val_or_test_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])

#     def __getitem__(self, idx):
#         image = io.imread(self.data[idx])

#         if len(image.shape) == 2:  # this is gray image
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#         img_resized = cv2.resize(image, (self.resize_width, self.resize_height), interpolation=cv2.INTER_CUBIC)
#         if self.mode == 'train':
#             return self.train_transform(img_resized), torch.tensor(self.target[idx]-1, dtype=torch.long)
#         elif self.mode == 'val' or self.mode == 'test':
#             return self.val_or_test_transform(img_resized), torch.tensor(self.target[idx] - 1, dtype=torch.long)

#     def __len__(self):
#         return len(self.data)


# class CarsDataLoader(BaseDataLoader):
#     """
#     Cars data loading
#     """
#     def __init__(self, mode, data_dir, metas, batch_size, resize_width,
#                  resize_height, shuffle=True, validation_split=0.0,
#                  num_workers=1, limit=None):

#         self.dataset = CarsDataset(mode, data_dir, metas, resize_width, resize_height, limit)

#         super(CarsDataLoader, self).__init__(self.dataset, batch_size, shuffle,
#                                              validation_split, num_workers)


class RandomGaussianNoise:
    def __init__(self, probability=0.15, standard_deviation=0.01):
        self.probability = probability
        self.standard_deviation = standard_deviation

    def __call__(self, tensor):
        if torch.rand(()) < self.probability:
            noise = torch.randn_like(tensor) * self.standard_deviation
            return (tensor + noise).clamp(0.0, 1.0)
        return tensor


class BrainMRIDataLoader(DataLoader):
    """Load the preprocessed brain MRI folder splits created by data_analysis.ipynb."""

    def __init__(
        self,
        mode,
        data_dir,
        batch_size,
        shuffle=False,
        validation_split=0.0,
        num_workers=0,
        weighted_sampling=False,
    ):
        if mode not in {"train", "validation", "test"}:
            raise ValueError("BrainMRIDataLoader mode must be train, validation, or test.")

        self.data_dir = Path(data_dir)
        normalization_path = self.data_dir / "normalization.json"
        if not normalization_path.is_file():
            raise FileNotFoundError(
                f"Missing normalization statistics: {normalization_path}. "
                "Run all preprocessing cells in data_processing/data_analysis.ipynb first."
            )
        with normalization_path.open("r", encoding="utf-8") as file:
            normalization = json.load(file)

        mean = normalization.get("mean")
        std = normalization.get("std")
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or len(mean) != 3
            or len(std) != 3
            or any(value <= 0 for value in std)
        ):
            raise ValueError(f"Invalid three-channel normalization values in {normalization_path}.")

        normalize = transforms.Normalize(mean=mean, std=std)
        self.train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                fill=0,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            RandomGaussianNoise(),
            normalize,
        ])
        self.evaluation_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        split_path = self.data_dir / mode
        if not split_path.is_dir():
            raise FileNotFoundError(f"Dataset split directory does not exist: {split_path}")
        transform = self.train_transform if mode == "train" else self.evaluation_transform
        self.dataset = ImageFolder(split_path, transform=transform)
        if len(self.dataset.classes) != 3:
            raise ValueError(
                f"Expected exactly 3 class folders under {split_path}; "
                f"found {self.dataset.classes}."
            )

        self.validation_dataset = None
        if mode == "train":
            validation_path = self.data_dir / "validation"
            if not validation_path.is_dir():
                raise FileNotFoundError(f"Validation split directory does not exist: {validation_path}")
            self.validation_dataset = ImageFolder(
                validation_path,
                transform=self.evaluation_transform,
            )
            if self.validation_dataset.classes != self.dataset.classes:
                raise ValueError(
                    "Training and validation class folders differ: "
                    f"{self.dataset.classes} != {self.validation_dataset.classes}"
                )

        self.n_samples = len(self.dataset)
        self.batch_idx = 0
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.class_to_idx = self.dataset.class_to_idx
        sampler = None
        if mode == "train" and weighted_sampling:
            class_counts = np.bincount(self.dataset.targets, minlength=len(self.dataset.classes))
            sample_weights = [1.0 / class_counts[target] for target in self.dataset.targets]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )

        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def split_validation(self):
        if self.validation_dataset is None:
            return None
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
