import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_data_loaders(df, expression_n, dmg_mut_n, protein_n, hot_mut_n, meta_n, RNA_n, CN_n, test_portion=0.1):
    """
    This function gets merged dataframe (or merged and grouped dataframe) and returns train, validation, test data loader.
    Other parameter indicates the columns of each omics data
    """
    # re-separate 
    expression = df[expression_n[1:]]
    damaging_mutation = df[dmg_mut_n[1:]]
    protein = df[protein_n[1:]]
    hotspot_mutation = df[hot_mut_n[1:]]
    metabolomic = df[meta_n[1:]]
    miRNA = df[RNA_n[1:]]
    copy_number = df[CN_n[1:]]
    target = df[["AUC", "IC50"]]
    cls = df["CLS"]
    #print(cls)
    # convert to tensor
    expression_tensor = torch.tensor(expression.to_numpy(dtype=np.float32))
    damaging_mutation_tensor = torch.tensor(damaging_mutation.to_numpy(dtype=np.float32))
    protein_tensor = torch.tensor(protein.to_numpy(dtype=np.float32))
    hotspot_mutation_tensor = torch.tensor(hotspot_mutation.to_numpy(dtype=np.float32))
    metabolomic_tensor = torch.tensor(metabolomic.to_numpy(dtype=np.float32))
    miRNA_tensor = torch.tensor(miRNA.to_numpy(dtype=np.float32))
    copy_number_tensor = torch.tensor(copy_number.to_numpy(dtype=np.float32))
    target_tensor = torch.tensor(target.to_numpy(dtype=np.float32))
    #cls_tensor = torch.tensor(cls)
    cls_array = np.stack(cls.to_numpy())
    cls_tensor = torch.tensor(cls_array, dtype=torch.float32)

    # train, validation, test split
    train_smiles, test_smiles, train_expression, test_expression, train_damaging_mutation, test_damaging_mutation, train_protein, test_protein, train_hotspot_mutation, test_hotspot_mutation, train_metabolomic, test_metabolomic, train_miRNA, test_miRNA, train_copy_number, test_copy_number, train_y, test_y = train_test_split(
        cls_tensor, expression_tensor, damaging_mutation_tensor, protein_tensor, hotspot_mutation_tensor, metabolomic_tensor, miRNA_tensor, copy_number_tensor, target_tensor, test_size=test_portion, random_state=0)

    train_smiles, val_smiles, train_expression, val_expression, train_damaging_mutation, val_damaging_mutation, train_protein, val_protein, train_hotspot_mutation, val_hotspot_mutation, train_metabolomic, val_metabolomic, train_miRNA, val_miRNA, train_copy_number, val_copy_number, train_y, val_y = train_test_split(
        train_smiles, train_expression, train_damaging_mutation, train_protein, train_hotspot_mutation, train_metabolomic, train_miRNA, train_copy_number, train_y, test_size=test_portion, random_state=0)

    """print("train_smiles shape =", train_smiles.shape)
    print("val_smiles shape =", val_smiles.shape)
    print("test_smiles shape =", test_smiles.shape)
    print("train_expression shape =", train_expression.shape)
    print("val_expression shape =", val_expression.shape)
    print("test_expression shape =", test_expression.shape)
    print("train_damaging_mutation shape =", train_damaging_mutation.shape)
    print("val_damaging_mutation shape =", val_damaging_mutation.shape)
    print("test_damaging_mutation shape =", test_damaging_mutation.shape)
    print("train_protein shape =", train_protein.shape)
    print("val_protein shape =", val_protein.shape)
    print("test_protein shape =", test_protein.shape)
    print("train_hotspot_mutation shape =", train_hotspot_mutation.shape)
    print("val_hotspot_mutation shape =", val_hotspot_mutation.shape)
    print("test_hotspot_mutation shape =", test_hotspot_mutation.shape)
    print("train_metabolomic shape =", train_metabolomic.shape)
    print("val_metabolomic shape =", val_metabolomic.shape)
    print("test_metabolomic shape =", test_metabolomic.shape)
    print("train_miRNA shape =", train_miRNA.shape)
    print("val_miRNA shape =", val_miRNA.shape)
    print("test_miRNA shape =", test_miRNA.shape)
    print("train_copy_number shape =", train_copy_number.shape)
    print("val_copy_number shape =", val_copy_number.shape)
    print("test_copy_number shape =", test_copy_number.shape)
    print("train_y shape =", train_y.shape)
    print("val_y shape =", val_y.shape)
    print("test_y shape =", test_y.shape)"""

    train_dataset = TensorDataset(train_smiles, train_expression, train_damaging_mutation, train_protein, train_hotspot_mutation, train_metabolomic, train_miRNA, train_copy_number, train_y)
    val_dataset = TensorDataset(val_smiles, val_expression, val_damaging_mutation, val_protein, val_hotspot_mutation, val_metabolomic, val_miRNA, val_copy_number, val_y)
    test_dataset = TensorDataset(test_smiles, test_expression, test_damaging_mutation, test_protein, test_hotspot_mutation, test_metabolomic, test_miRNA, test_copy_number, test_y)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    """print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Validation Dataset: {len(val_dataset)} samples")
    print(f"Test Dataset: {len(test_dataset)} samples")"""

    return train_loader, val_loader, test_loader

def test_model(model, test_loader, device, title=None):
    model.to(device)
    model.eval()
    ic50_preds, ic50_targets = [], []
    auc_preds, auc_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            smiles_list, expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data, targets = batch
            omics_data = [expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data]
            ic50_true, auc_true = targets[:, 0], targets[:, 1]
            
            smiles_list = smiles_list.to(device)
            omics_data = [data.to(device) for data in omics_data]
            ic50_true = ic50_true.to(device)
            auc_true = auc_true.to(device)

            ic50_pred, auc_pred = model(smiles_list, omics_data)

            ic50_mask = ~torch.isnan(ic50_true)
            auc_mask = ~torch.isnan(auc_true)

            if ic50_mask.sum() > 0:
                ic50_preds.append(ic50_pred[ic50_mask].cpu().numpy())
                ic50_targets.append(ic50_true[ic50_mask].cpu().numpy())
            if auc_mask.sum() > 0:
                auc_preds.append(auc_pred[auc_mask].cpu().numpy())
                auc_targets.append(auc_true[auc_mask].cpu().numpy())

        # list -> numpy
        ic50_preds = np.concatenate(ic50_preds) if ic50_preds else np.array([])
        ic50_targets = np.concatenate(ic50_targets) if ic50_targets else np.array([])
        auc_preds = np.concatenate(auc_preds) if auc_preds else np.array([])
        auc_targets = np.concatenate(auc_targets) if auc_targets else np.array([])

        # evaluation
        metrics = {}
        if len(ic50_preds) > 0:
            metrics["IC50_MSE"] = mean_squared_error(ic50_targets, ic50_preds)
            metrics["IC50_MAE"] = mean_absolute_error(ic50_targets, ic50_preds)
            metrics["IC50_R2"] = r2_score(ic50_targets, ic50_preds)
        else:
            metrics["IC50_MSE"] = -100
            metrics["IC50_MAE"] = -100
            metrics["IC50_R2"] = -100
        if len(auc_preds) > 0:
            metrics["AUC_MSE"] = mean_squared_error(auc_targets, auc_preds)
            metrics["AUC_MAE"] = mean_absolute_error(auc_targets, auc_preds)
            metrics["AUC_R2"] = r2_score(auc_targets, auc_preds)
        else:
            metrics["AUC_MSE"] = -100
            metrics["AUC_MAE"] = -100
            metrics["AUC_R2"] = -100

        print("\n **Test Results:**")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # IC50 Scatter Plot
        if len(ic50_preds) > 0:
            plt.figure(figsize=(6,6))
            plt.scatter(ic50_targets, ic50_preds, alpha=0.5, label="IC50 Predictions")
            # y = x 
            min_val = min(np.min(ic50_targets), np.min(ic50_preds))
            max_val = max(np.max(ic50_targets), np.max(ic50_preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
            plt.xlabel("IC50 Target")
            plt.ylabel("IC50 Prediction")
            if title:
                plt.title(title)
            else:
                plt.title("IC50 Scatter Plot")
            # R2
            plt.text(0.05, 0.90, f"R2 = {metrics['IC50_R2']:.4f}", transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            plt.legend(loc="lower right")
            plt.show()
        
        # AUC Scatter Plot
        if len(auc_preds) > 0:
            plt.figure(figsize=(6,6))
            plt.scatter(auc_targets, auc_preds, alpha=0.5, label="AUC Predictions")
            min_val = min(np.min(auc_targets), np.min(auc_preds))
            max_val = max(np.max(auc_targets), np.max(auc_preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
            plt.xlabel("AUC Target")
            plt.ylabel("AUC Prediction")
            if title:
                plt.title(title)
            else:
                plt.title("AUC Scatter Plot")
            plt.text(0.05, 0.90, f"R2 = {metrics['AUC_R2']:.4f}", transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            plt.legend(loc="lower right")
            plt.show()
        
        return metrics
    
def test_missing_pairs(model, test_loader, device):
    model.to(device)
    model.eval()
    ic50_preds = []
    auc_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            smiles_list, expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data = batch
            omics_data = [expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data]
      
            smiles_list = smiles_list.to(device)
            omics_data = [data.to(device) for data in omics_data]

            ic50_pred, auc_pred = model(smiles_list, omics_data)

            ic50_preds.append(ic50_pred.cpu().numpy())

            auc_preds.append(auc_pred.cpu().numpy())

        # list -> numpy
        ic50_preds = np.concatenate(ic50_preds) if ic50_preds else np.array([])
        auc_preds = np.concatenate(auc_preds) if auc_preds else np.array([])

        return ic50_preds, auc_preds