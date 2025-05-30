#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hamzaoui
"""
import torch
from tqdm import tqdm
import numpy as np
from utils.Evaluation_metrics import eval_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_validation(model,val_loader,fnet_preds,mode="test",all_labels=[],criterion=None,val_loss=0,iso_preds=[],all_preds=[],iso_forest=None):
    """
    Performs validation or testing for one epoch, supporting multiple evaluation modes.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation or test set.
        fnet_preds (list): List to store FaceNet-based predictions.
        mode (str): Evaluation mode. "Val" for validation, "Xval" for cross-validation with anomaly detection, "test" for test mode.
        all_labels (list): List to store true labels (used in validation).
        criterion (callable, optional): Loss function (used in validation).
        val_loss (float): Accumulated validation loss.
        iso_preds (list): List to store Isolation Forest predictions (used in "Xval" mode).
        all_preds (list): List to store final predictions (used in "Xval" mode).
        iso_forest (IsolationForest, optional): Isolation Forest model for anomaly detection.

    Returns:
        val_loss (float): Total validation loss accumulated over the epoch.
    """
    with torch.no_grad():
            for images, labels in tqdm(val_loader):
                outputs = model(images.to(device))   
                preds = (torch.sigmoid(outputs) > 0.5).float()
                fnet_preds.extend(preds.cpu().numpy().flatten())
                if mode=="Val":
                    all_labels.extend(labels.cpu().numpy())
                    loss = criterion(outputs, labels.float().unsqueeze(1).to(device))
                    val_loss += loss.item() * images.size(0)
                if mode=="Xval":
                    val_loss += loss.item() * images.size(0)
                    features_128 = model.get_features(images.to(device), which='128').cpu().numpy()
                    iso_forest.fit(features_128)
                    i_preds = iso_forest.predict(features_128)
                    i_preds = (i_preds + 1) / 2  
                    iso_preds.extend(i_preds.flatten())
    if mode=="Xval":
        all_preds.extend(list(((np.array(iso_preds) + np.array(fnet_preds))/2 >0.5).astype(int)))
    return val_loss
                

def epoch_train(model,train_loader,criterion,running_loss,optimizer):
    """
    Performs one training epoch for the given model.

    Args:
        model (torch.nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader providing training batches.
        criterion (callable): Loss function to optimize.
        running_loss (float): Accumulated loss for the epoch.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.

    Returns:
        running_loss (float): Total loss accumulated over the epoch.
    """
    for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * images.size(0)
    return running_loss




def fine_tune_Xval(train_loader,val_loader,model,optimizer,criterion,num_epochs,results,iso_forest,fold):
    """
    Fine-tunes the model using cross-validation, tracks metrics, and applies learning rate scheduling.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): Model to be fine-tuned.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (callable): Loss function.
        num_epochs (int): Number of training epochs.
        results (dict): Dictionary to store metrics and results.
        iso_forest (IsolationForest): Isolation Forest model for anomaly detection.
        fold (int): Current fold number for cross-validation.

    Behavior:
        - Trains and validates the model for the specified number of epochs.
        - Uses ReduceLROnPlateau scheduler to reduce the learning rate when validation loss plateaus[1][3].
        - Collects and stores various metrics (balanced accuracy, HTER, validation loss) for FaceNet and Isolation Forest classifiers.
        - Prints summary statistics for each epoch and fold.
    """
    # Initialize learning rate scheduler to reduce LR when validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    for epoch in range(num_epochs):
        print("fine tuning")
        model.train()
        running_loss = 0.0
        running_loss = epoch_train(model,train_loader,criterion,running_loss,optimizer)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        print("validation")
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        fnet_preds = [] 
        iso_preds  = []
        
        val_loss = epoch_validation(model,val_loader,fnet_preds,"Xval",all_labels,criterion,val_loss,iso_preds,all_preds,iso_forest)
        
        results['fold'].append(fold)
        balanced_acc,HTER = eval_metrics(all_labels,fnet_preds)
        results['recap']['balanced_acc_FNET'].append(balanced_acc)
        results['recap']['val_loss'].append(val_loss/len(val_loader.dataset))
        results['recap']['HTER_FNET'].append(HTER)
        scheduler.step(val_loss)
        if iso_preds:
            balanced_acc,HTER = eval_metrics(all_labels,iso_preds)        
            results['recap']['balanced_acc_ISO'].append(balanced_acc)
            results['recap']['HTER_ISO'].append(HTER)

            balanced_acc,HTER = eval_metrics(all_labels,all_preds)        
            results['recap']['balanced_acc'].append(balanced_acc)
            results['recap']['HTER'].append(HTER)

        
        print(f"Fold {fold+1} | Balanced Acc: {results['recap']['balanced_acc'][-1]:.4f} | HTER: {results['recap']['HTER'][-1]:.4f}")
        print(f"FNET | Balanced Acc: {results['recap']['balanced_acc_FNET'][-1]:.4f} | | HTER: {results['recap']['HTER_FNET'][-1]:.4f}| Val Loss: {results['recap']['val_loss'][-1]:.4f}")
        print(f"ISOF | Balanced Acc: {results['recap']['balanced_acc_ISO'][-1]:.4f} | | HTER: {results['recap']['HTER_ISO'][-1]:.4f}")


def fine_tune_test(train_loader,test_loader,model,optimizer,criterion,num_epochs,results):
    """
    Fine-tunes the model on the training set and evaluates on the test set after each epoch (starting from the 4th).

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        model (torch.nn.Module): Model to be trained and evaluated.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (callable): Loss function.
        num_epochs (int): Number of training epochs.
        results (str): Path prefix for saving prediction results per epoch.

    Behavior:
        - Trains the model for the specified number of epochs.
        - Prints training loss after each epoch.
        - Starting from the 4th epoch, evaluates the model on the test set after each epoch.
        - Saves the test set predictions for each evaluated epoch to a text file.
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_loss = epoch_train(model,train_loader,criterion,running_loss,optimizer)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        if epoch>3: # concluded from cross-validation to monitor values starting 4th epoch
        
            model.eval()
            fnet_preds = [] 
            epoch_validation(model,test_loader,fnet_preds)
            np.savetxt(f"{results}{epoch}.txt", np.array(fnet_preds, dtype=int), fmt="%d")
            
            
            
def fine_tune_val_test(train_loader,val_loader,test_loader,model,optimizer,criterion,num_epochs,results):
    """
    Fine-tunes the model using training and validation sets, tracks the best model based on HTER, 
    and evaluates on the test set when a new best model is found.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        model (torch.nn.Module): Model to be trained and evaluated.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (callable): Loss function.
        num_epochs (int): Total number of training epochs.
        results (str): Path prefix for saving models and prediction results.

    Behavior:
        - Trains the model starting from the 4th epoch (based on prior cross-validation).
        - After each epoch, evaluates on the validation set and computes Balanced Accuracy and HTER.
        - If a new best HTER is achieved (lower than previous best), saves the model and evaluates on the test set.
        - Saves test set predictions for the best model.
        - Uses ReduceLROnPlateau scheduler to adjust learning rate based on validation loss.
        - Prints summary metrics and learning rate after each epoch.
    """
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    best_hter = 0.15
    for epoch in range(4,num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_loss = epoch_train(model,train_loader,criterion,running_loss,optimizer)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        all_labels = []
        fnet_preds = [] 
    
        val_loss = epoch_validation(model,val_loader,fnet_preds,"Val",all_labels,criterion,val_loss)
        balanced_acc,HTER = eval_metrics(all_labels,fnet_preds)
        print(f"FNET | Balanced Acc: {balanced_acc:.4f} | | HTER: {HTER:.4f}| Val Loss: {val_loss/len(val_loader.dataset):.4f}")
        
        if epoch>3: # concluded from cross-validation to monitor values starting 4th epoch
            if HTER< best_hter:
                best_hter = HTER
                fnet_preds = [] 
                torch.save(model.state_dict(), f'{results}best_model_epoch{epoch+1}_HTER{best_hter:.4f}.pt')
                print(f"Saved new best model at epoch {epoch+1} with HTER {best_hter:.4f}")
                epoch_validation(model,test_loader,fnet_preds)
                np.savetxt(f"{results}label_validation_{epoch+1}.txt", np.array(fnet_preds, dtype=int), fmt="%d")
        
        scheduler.step(val_loss)
        print(f"last learning rate: {scheduler.get_last_lr()[-1]}")
        

