import numpy as np
import os
import pandas as pd
from PIL import Image
from random import shuffle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, List, Tuple


class AddNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std
        
    def __call__(self, x: torch.Tensor):
        return x + torch.randn(x.size()) * self.std


class MILDataset(Dataset):
    def __init__(self, tile_dirs: List[str], labels: List[int], num_tiles_to_sample: int, train: bool = True) -> None:
        super().__init__()

        num_tiles = set([len(os.listdir(tile_dir)) for tile_dir in tile_dirs]) 
        if len(num_tiles) != 1:
            raise ValueError("Need identical number of tiles per patient. What I got is:", num_tiles)
        self.num_tiles_per_patient = num_tiles.pop()
        self.num_tiles_to_sample = num_tiles_to_sample
        self.track = 0
        self.handle = 0
        
        self.data = [
            [np.array(Image.open(os.path.join(tile_dir, tile))) for tile in os.listdir(tile_dir)]
            for tile_dir in tile_dirs
        ]
        self.labels = labels
        transform = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        if train:
            transform.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                AddNoise(),
            ])
        self.transform = transforms.Compose(transform)
        self.shuffle()

    def shuffle(self) -> None:
        print("Shuffling...")
        for i in range(len(self)):
            shuffle(self.data[i])

    def transform(self, x: np.ndarray):
        return self.transform(x)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], int]:
        features = torch.stack([
            self.transform(item)  
            for item in self.data[idx][self.handle * self.num_tiles_to_sample: (self.handle + 1) * self.num_tiles_to_sample]
        ])
        label = self.labels[idx]
        self.track += 1
        if self.track == len(self):
            self.handle += 1
            self.track = 0
        if self.handle == self.num_tiles_per_patient // self.num_tiles_to_sample:
            self.shuffle()
            self.handle = 0
            self.track = 0
        return features, label
    

def get_image_ids(orthopedia_dir: str, class_: str, split: str) -> List[str]:
    return pd.read_csv(os.path.join(orthopedia_dir, "csv_files", f"{class_}_{split}.csv"))["imageID"].astype(int).astype(str).to_list()


def get_fns(orthopedia_dir: str, class_: str, split: str) -> List[str]:
    return pd.read_csv(os.path.join(orthopedia_dir, "csv_files", f"{class_}_{split}.csv"))["Fallnummer"].astype(str).to_list()


def create_mil_datasets(orthopedia_dir: str, num_tiles_to_sample: int = 50) -> Dict[str, MILDataset]:
    splits = {"train", "val", "test"}
    tile_dirs = {split: list() for split in splits}
    labels = {split: list() for split in splits}
    fallnummers = {split: get_fns(orthopedia_dir, "infect", split) + get_fns(orthopedia_dir, "noinfect", split) for split in splits}
    for key_folder in ["infect", "noinfect"]:
        key_path = os.path.join(orthopedia_dir, f"{key_folder}_tiles")
        for fn_folder in os.listdir(key_path):
            mapping = [split for split in splits if fn_folder in fallnummers[split]]
            if len(mapping) == 1:
                split = mapping[0]
            else:
                raise ValueError
            fn_path = os.path.join(key_path, fn_folder)
            for tile_folder in os.listdir(fn_path):
                tile_dir = os.path.join(fn_path, tile_folder)
                if len(os.listdir(tile_dir)) == 0:
                    print(f"Directory {tile_dir} is empty, skipping...")
                    continue
                tile_dirs[split].append(tile_dir)
                labels[split].append(1 if key_folder == "infect" else 0)
    return {
        split: MILDataset(tile_dirs[split], labels[split], num_tiles_to_sample, train = split == "train")
        for split in splits
    }


def collate_fn(data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.stack([sample[0] for sample in data]).float(), torch.tensor([sample[1] for sample in data]).float()
