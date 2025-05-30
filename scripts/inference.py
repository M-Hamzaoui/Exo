#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamzaoui
"""


from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from utils.IO_utils import CustomDataset
from models.facenet import FaceNetWithReduction
from utils.train_validate_utils import epoch_validation
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

location = "" # where ml_exercise_therapanacea is stored

results=f'{location}/ml_exercise_therapanacea/label_val.txt'
weights=f'{location}/ml_exercise_therapanacea/label_validation_best_model.pt'

# =============================================================================
#                            Trasnforamtions 
# only resize and intensity normalization for validation and test
# =============================================================================


validation_transform = T.Compose([
    T.Resize((160,160)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
# =============================================================================
#                            Train dataset 
# read data, split train-val (LR), (train data unbalanced) => sampler, loaders
# =============================================================================

# =============================================================================
#                         Test dataset 
# =============================================================================

Test_dataset = CustomDataset(
    img_dir=f'{location}/ml_exercise_therapanacea/train_img',
    label_file='',
    transform=validation_transform,mode="v"
)

test_loader = DataLoader(Test_dataset, batch_size=32, shuffle=False)

# =============================================================================
#                         define model, optimizer and criterion 
# =============================================================================

model = FaceNetWithReduction(pretrained='vggface2').to(device)

model.load_state_dict(torch.load(weights))
model.eval()

fnet_preds = []
epoch_validation(model,test_loader,fnet_preds)
np.savetxt(results, np.array(fnet_preds, dtype=int), fmt="%d")
