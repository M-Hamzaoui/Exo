#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamzaoui
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
from utils.IO_utils import CustomDataset
from models.facenet import FaceNetWithReduction
from utils.train_validate_utils import fine_tune_test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

location = "" # where ml_exercise_therapanacea is stored
results=f'{location}/ml_exercise_therapanacea/label_validation_'
# =============================================================================
#                            Train dataset 
# define transformations, read data, (train data unbalanced) => sampler, loader
# =============================================================================
train_transform = T.Compose([
    T.Resize((160,160)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = CustomDataset(
    img_dir=f'{location}/ml_exercise_therapanacea/train_img',
    label_file=f'{location}/ml_exercise_therapanacea/label_train.txt',
    transform=train_transform,mode="t"
)

class_counts = train_dataset.labels['label'].value_counts().sort_index().values
# when trying to counter data imbalance using random sampling
# weights = 1./torch.tensor(class_counts, dtype=torch.float, device=device)
# samples_weights = weights[train_labels.values]
# sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)# sampler=sampler)

# =============================================================================
#                         Validation dataset 
# =============================================================================
validation_transform = T.Compose([
    T.Resize((160,160)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

test_dataset = CustomDataset(
    img_dir=f'{location}/ml_exercise_therapanacea/train_img',
    label_file='',
    transform=validation_transform,mode="v"
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =============================================================================
#                         define model, optimizer and criterion 
# =============================================================================

model = FaceNetWithReduction(pretrained='vggface2').to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Penalize error on minority class (0 here)
pos_weight = torch.tensor([class_counts[0]/class_counts[1]],device=device) 

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
num_epochs= 30

fine_tune_test(train_loader,test_loader,model,optimizer,criterion,num_epochs,results)
