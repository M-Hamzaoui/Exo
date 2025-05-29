#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:19:39 2025

@author: hamzaoui
"""

import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetWithReduction(nn.Module):
    def __init__(self, pretrained='vggface2', freeze_until='block8'):
        super().__init__()

        facenet = InceptionResnetV1(pretrained=pretrained, classify=True)
        facenet.logits = nn.Linear(facenet.logits.in_features, 128)

        for name, param in facenet.named_parameters():
            if any(key in name for key in ["block8", "last_linear", "last_bn", "logits"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.facenet = facenet
        self.embedding_reduce = nn.Linear(128, 4)
        self.actifin = nn.LeakyReLU(0.1)
        self.embedding_classifier = nn.Linear(4, 1)

    def forward(self, x):
        embedding = self.facenet(x)
        reduced = self.actifin(self.embedding_reduce(embedding))
        classified = self.embedding_classifier(reduced)
        return classified

    def get_features(self, x, which='128'):
        embedding = self.facenet(x)
        if which == '128':
            return embedding
        elif which == '4':
            return self.reduce(embedding)
        else:
            raise ValueError("which must be '128' or '4'")