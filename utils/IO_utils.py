#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:56:50 2025

@author: hamzaoui
"""

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path



class CustomDataset(Dataset):
    def __init__(self, img_dir, label_file, mode, transform=None):
        self.img_dir = Path(img_dir)
        self.mode = mode
        if mode == "t":
            self.labels = pd.read_csv(label_file, sep=' ', header=None, names=['label'])
        else:
            self.labels = None 
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_dir / f"{str(idx + 1).zfill(6)}.jpg"
        image = Image.open(img_path).convert('RGB')
        if self.mode == "t":
            label = self.labels.iloc[idx]['label']
        else:
            label = -1
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        if self.mode=="t":
            return len(self.labels)
        else:
            return 20000
