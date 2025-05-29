Test Setup and Model Development Overview

Objective:
Develop a binary classifier evaluated primarily using the Half Total Error Rate (HTER).

Dataset Observations

1/    Modality: RGB facial images, shape (3, 64, 64).

2/    Imbalance:

        Class 0: 12%

        Class 1: 88%

3/    Feature Type: Classification is based on facial features.

Methodology

1/    Feature Extraction:
      Leveraged a pretrained FaceNet model (InceptionResnetV1) trained on the VGGFace2 dataset to obtain facial embeddings.

2/    Classification Strategies:

        Fine-tuned the final layers of the pretrained FaceNet model.

        Built a lightweight Multi-Layer Perceptron (MLP) on top of the embeddings.

        Additionally, trained an Isolation Forest on the FaceNet embeddings for anomaly detection.

3/    Model Selection via Cross-Validation:

        Conducted stratified 5-fold cross-validation to assess stability.

        Results indicated superior performance of the MLP classifier over the Isolation Forest.

        Experimented with varying degrees of layer freezing during fine-tuning; initializing with pretrained weights consistently improved performance.

4/    Addressing Class Imbalance:

        Employed a weighted random sampler in the data loader to balance training batches.

        Applied weighted binary cross-entropy (BCE) loss, assigning higher weight to the minority class (class 0).

5/    Training and Evaluation Protocol:

        Adopted a 90%-10% train-validation split from the training data.

        Applied ReduceLROnPlateau to adjust learning rate dynamically based on validation performance.

        Used this strategy to identify the optimal stopping point (epoch) by monitoring HTER on the validation set.

6/    Final Testing:

        After selecting the best-performing model based on validation HTER, the final evaluation was conducted on the held-out test set.
