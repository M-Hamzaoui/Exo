#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:25:27 2025

@author: hamzaoui
"""
import torch
from tqdm import tqdm
import numpy as np
from utils.Evaluation_metrics import eval_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def epoch_validation(model,val_loader,fnet_preds,mode="test",all_labels=[],criterion=None,val_loss=0,iso_preds=[],all_preds=[],iso_forest=None):
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
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
                torch.save(model.state_dict(), f'{results}best_model_epoch{epoch+1}_HTER{best_hter:.4f}.pt')
                print(f"Saved new best model at epoch {epoch+1} with HTER {best_hter:.4f}")
                epoch_validation(model,test_loader,fnet_preds)
                np.savetxt(f"{results}label_validation_{epoch+1}.txt", np.array(fnet_preds, dtype=int), fmt="%d")
        
        scheduler.step(val_loss)
        print(f"last learning rate: {scheduler.get_last_lr()[-1]}")
        

