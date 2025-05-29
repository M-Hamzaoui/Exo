#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:19:39 2025

@author: hamzaoui
"""

import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetWithReduction(nn.Module):
    """
    Custom neural network for face classification based on a modified FaceNet (InceptionResnetV1) backbone.

    Structure:
        - Uses a pretrained InceptionResnetV1 model from facenet-pytorch as the backbone for feature extraction.
        - Replaces the final classification layer to output 128-dimensional embeddings.
        - Freezes all layers except for 'block8', 'last_linear', 'last_bn', and 'logits' to enable selective fine-tuning.
        - Adds a dimensionality reduction layer (128 -> 4) with LeakyReLU activation.
        - Adds a final classification layer (4 -> 1) for binary output.

    Args:
        pretrained (str): Name of the pretrained weights to use ('vggface2' by default).
        freeze_until (str): Layer name from which to allow gradient updates (default: 'block8').

    Methods:
        forward(x): Computes the forward pass and returns the classification output.
        get_features(x, which): Returns either the 128-dimensional or reduced 4-dimensional embedding. (used to fit the isolation forest model, which was disrecarded in the final aproach)
    """
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
