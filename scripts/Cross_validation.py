#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:39:38 2025

@author: hamzaoui
"""

from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import numpy as np
import torch.nn as nn
from IO_utils import CustomDataset
from facenet import FaceNetWithReduction
from sklearn.ensemble import IsolationForest
from train_validate import fine_tune_Xval


train_transform = T.Compose([
    T.Resize(160),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = CustomDataset(
    img_dir='/home/hamzaoui/Downloads/ml_exercise_therapanacea/train_img',
    label_file='/home/hamzaoui/Downloads/ml_exercise_therapanacea/label_train.txt',
    transform=train_transform,mode="t"
)

validation_transform = T.Compose([
    T.Resize((160,160)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

kfolds = 5
kfold = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

results = {'fold': [] , 'recap': {'balanced_acc': [], 'val_loss': [], 'HTER': [],'balanced_acc_FNET': [], 'HTER_FNET': [],'balanced_acc_ISO': [], 'HTER_ISO': []}}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset.labels['label'], dataset.labels['label'])):
    print(f"\nFold {fold+1}/{kfolds}")
    model = FaceNetWithReduction(pretrained='vggface2').to(device)
    iso_forest = IsolationForest(contamination=0.12, random_state=42)   # trying another model unsupervised learning depending on the embedding given by the facenet

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_labels = dataset.labels.iloc[train_idx]['label']
    class_counts = train_labels.value_counts().sort_index().values
    pos_weight = torch.tensor([class_counts[0]/class_counts[1]],device=device)  # Penalize class 0 more
    
    weights = 1./torch.tensor(class_counts, dtype=torch.float, device=device)
    samples_weights = weights[train_labels.values]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(CustomDataset(img_dir='/home/hamzaoui/Downloads/ml_exercise_therapanacea/train_img',label_file='/home/hamzaoui/Downloads/ml_exercise_therapanacea/label_train.txt',transform=train_transform,mode="t"), val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler) 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    num_epochs= 25
    fine_tune_Xval(train_loader,val_loader,model,optimizer,criterion,num_epochs,results,iso_forest,fold)
        

print(f"\nFinal Results: Avg Balanced Acc = {np.mean(results['balanced_acc']):.4f}")
    
