# data preprocessing (data_type = "", each data_type will have each preprocess type)
# - IC50, AUC: log transform
# training
# evaluating
# plotting
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from MyModels import *
import os

data_directory = '../data/'
file_names = [
    'Batch_corrected_Expression_Public_24Q4_subsetted.csv',
    'Damaging_Mutations_subsetted.csv', 
    'Harmonized_RPPA_CCLE_subsetted.csv',
    'Hotspot_Mutations_subsetted.csv', 
    'IC50_AUC_merged.csv', 
    'Metabolomics_subsetted.csv',
    'miRNA_Expression_subsetted.csv',
    'Omics_Absolute_CN_Gene_Public_24Q4_subsetted.csv'
    ]


def get_CLS_IC50_AUC():
    # get cls tokens
    smiles = pd.read_csv("../data/drugID_name_pubchem_smiles.csv") # this file is uploaded in git

    # load ChemBERTa tokenizer and model
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # tokenize
    tokens = tokenizer(smiles["smiles"].tolist(), padding=True, truncation=True, return_tensors="pt")

    # embedded tokenized SMILES
    with torch.no_grad():
        outputs = model(**tokens)

    # CLS token (for now, we will only use CLS token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_dim)

    cls_embeddings_np = cls_embeddings.cpu().numpy()

    smiles["CLS"] = list(cls_embeddings_np)

    # get IC50 and AUC
    IC50_AUC = pd.read_csv("../data/IC50_AUC_merged.csv")
    IC50_AUC = IC50_AUC.rename(columns={"Unnamed: 0": "CellLineID"})

    melted = IC50_AUC.melt(
        id_vars=["CellLineID"],  # fixed column
        var_name="metric_drug",  # column names that will be one single column
        value_name="value"       
    )

    # ex: "Drug sensitivity AUC (Sanger GDSC2) CAMPTOTHECIN (GDSC2:1003)" -> (AUC), (CAMPTOTHECIN), (1003)
    pattern = r"(AUC|IC50).*?\((Sanger GDSC2)\)\s+(.*?)\s+\(GDSC2:(\d+)\)"

    # (AUC|IC50) -> 1) AUC or IC50
    # (.*?)      -> 3) Drug name (minimum match)
    # (\d+)      -> 4) drug number from GDSC2:XXXX

    melted[["Metric", "_sanger", "DrugName", "DrugNumber"]] = melted["metric_drug"].str.extract(pattern)
    new_melted = melted[['CellLineID', 'value', 'Metric','DrugNumber']]
    
    final_IC50_AUC = (
        new_melted
        .pivot(index=["CellLineID", "DrugNumber"], 
            columns="Metric", 
            values="value")
        #.reset_index()
    )
    final_IC50_AUC = final_IC50_AUC.dropna(subset=['AUC', 'IC50'], how='all') # drop the row that AUC and IC50 are both NAN
    final_IC50_AUC.columns.name = None
    IC50_AUC_final = final_IC50_AUC.reset_index()

    # merge IC50, AUC, and SMILES (CLS)
    IC50_AUC_final["DrugNumber"] = IC50_AUC_final["DrugNumber"].astype(int)
    smiles["drugID"] = smiles["drugID"].astype(int)
    df_merged = pd.merge(IC50_AUC_final, smiles, left_on='DrugNumber', right_on='drugID', how='inner')

    ic50_data = df_merged[['CellLineID', 'DrugNumber', 'smiles', 'CLS', 'IC50']].copy()
    ic50_data['AUC'] = None # each row will have either AUC or IC50 value. 
    ic50_data = ic50_data.rename(columns={'IC50': 'IC50'})

    auc_data = df_merged[['CellLineID', 'DrugNumber', 'smiles', 'CLS', 'AUC']].copy()
    auc_data['IC50'] = None 
    auc_data = auc_data.rename(columns={'AUC': 'AUC'})

    final_df = pd.concat([ic50_data, auc_data])
    final_df = final_df.sort_values(by=['CellLineID', 'DrugNumber']).reset_index(drop=True)
    df = final_df.dropna(subset=['AUC', 'IC50'], how='all').reset_index(drop=True)

    #df = pd.read_csv("../data/final_CLS_IC50_AUC.csv")
    epsilon = 1e-10  # prevent zero division
    y = df[["AUC", "IC50"]]
    #print(y.describe())
    y_array = y.to_numpy(dtype=np.float32)
    y_array[:, 0] = np.log10(np.where(y_array[:, 0] == 0, epsilon, y_array[:, 0])) 
    #y_array[:, 1] = np.log(np.where(y_array[:, 1] == 0, epsilon, y_array[:, 1]))
    df["AUC"] = y_array[:, 0]

    scaler_IC50 = MinMaxScaler()
    y_array[:, 1] = scaler_IC50.fit_transform(y_array[:, 1].reshape(-1, 1)).flatten()
    df["IC50"] = y_array[:, 1]
    #print(df)
    return df

def get_omics_data(omics_name): # read omics data file and preprocess
    if omics_name == "expression":
        batch_corrected_expression = pd.read_csv(f'{data_directory}{file_names[0]}')
        batch_corrected_expression = batch_corrected_expression.rename(columns={batch_corrected_expression.columns[0]: "CellLineID"})
        batch_corrected_expression.columns = ['CellLineID'] + [f"1_{col}" for col in batch_corrected_expression.columns[1:]]

        expression_cnv = batch_corrected_expression.iloc[:, 1:]
        pca_expression = PCA(n_components=500)
        expression_pca = pca_expression.fit_transform(expression_cnv)
        expression_pca_df = pd.DataFrame(expression_pca)
        expression_pca_df.insert(0, "CellLineID", batch_corrected_expression["CellLineID"].values)
        expression_pca_df.columns = ['CellLineID'] + [f"1_{col}" for col in expression_pca_df.columns[1:]]
        df = expression_pca_df

    elif omics_name == "damaging_mutation":
        damaging_mutations = pd.read_csv(f'{data_directory}{file_names[1]}')
        damaging_mutations = damaging_mutations.rename(columns={damaging_mutations.columns[0]: "CellLineID"})
        damaging_mutations.columns = ['CellLineID'] + [f"2_{col}" for col in damaging_mutations.columns[1:]]

        dmg_mut = damaging_mutations.drop(columns="CellLineID")
        selector_mut = VarianceThreshold(threshold=0.01)
        dmg_mut_reduced = selector_mut.fit_transform(dmg_mut)
        selected_columns_mut = dmg_mut.columns[selector_mut.get_support()]
        scaler_mut = RobustScaler()
        dmg_mut_reduced = scaler_mut.fit_transform(dmg_mut_reduced) 
        dmg_mut_reduced_df = pd.DataFrame(dmg_mut_reduced, columns=selected_columns_mut)
        dmg_mut_reduced_df = pd.concat([damaging_mutations["CellLineID"], dmg_mut_reduced_df], axis=1)
        df = dmg_mut_reduced_df

    elif omics_name == "protein":
        harmonized_RPPA = pd.read_csv(f'{data_directory}{file_names[2]}')
        harmonized_RPPA = harmonized_RPPA.rename(columns={harmonized_RPPA.columns[0]: "CellLineID"})
        harmonized_RPPA.columns = ['CellLineID'] + [f"3_{col}" for col in harmonized_RPPA.columns[1:]]

        prot = harmonized_RPPA.drop(columns="CellLineID")
        scaler_protein = StandardScaler()
        protein_scaled = scaler_protein.fit_transform(prot)
        protein_scaled_df = pd.DataFrame(protein_scaled)
        protein_scaled_df.insert(0, "CellLineID", harmonized_RPPA["CellLineID"].values)
        protein_scaled_df.columns = ['CellLineID'] + [f"3_{col}" for col in protein_scaled_df.columns[1:]]
        df = protein_scaled_df

    elif omics_name == "hotspot_mutation":
        hotspot_mutations = pd.read_csv(f'{data_directory}{file_names[3]}')
        hotspot_mutations = hotspot_mutations.rename(columns={hotspot_mutations.columns[0]: "CellLineID"})
        hotspot_mutations.columns = ['CellLineID'] + [f"4_{col}" for col in hotspot_mutations.columns[1:]]

        hot_mut = hotspot_mutations.drop(columns="CellLineID")
        scaler_hotspot = StandardScaler()
        hotspot_scaled = scaler_hotspot.fit_transform(hot_mut)
        hotspot_scaled_df = pd.DataFrame(hotspot_scaled)
        hotspot_scaled_df.insert(0, "CellLineID", hotspot_mutations["CellLineID"].values)
        hotspot_scaled_df.columns = ['CellLineID'] + [f"4_{col}" for col in hotspot_scaled_df.columns[1:]]
        df = hotspot_scaled_df

    elif omics_name == "metabolomics":
        metabolomics = pd.read_csv(f'{data_directory}{file_names[5]}')
        metabolomics = metabolomics.rename(columns={metabolomics.columns[0]: "CellLineID"})
        metabolomics.columns = ['CellLineID'] + [f"5_{col}" for col in metabolomics.columns[1:]]

        meta = metabolomics.drop(columns="CellLineID")
        scaler_metabolomics = StandardScaler()
        metabolomics_scaled = scaler_metabolomics.fit_transform(meta)
        metabolomics_scaled_df = pd.DataFrame(metabolomics_scaled)
        metabolomics_scaled_df.insert(0, "CellLineID", metabolomics["CellLineID"].values)
        metabolomics_scaled_df.columns = ['CellLineID'] + [f"5_{col}" for col in metabolomics_scaled_df.columns[1:]]
        df = metabolomics_scaled_df

    elif omics_name == "miRNA":
        miRNA_expression = pd.read_csv(f'{data_directory}{file_names[6]}')
        miRNA_expression = miRNA_expression.rename(columns={miRNA_expression.columns[0]: "CellLineID"})
        miRNA_expression.columns = ['CellLineID'] + [f"6_{col}" for col in miRNA_expression.columns[1:]]

        rna = miRNA_expression.drop(columns="CellLineID")
        scaler_miRNA = StandardScaler()
        miRNA_scaled = scaler_miRNA.fit_transform(rna)
        miRNA_scaled_df = pd.DataFrame(miRNA_scaled)
        miRNA_scaled_df.insert(0, "CellLineID", miRNA_expression["CellLineID"].values)
        miRNA_scaled_df.columns = ['CellLineID'] + [f"6_{col}" for col in miRNA_scaled_df.columns[1:]]
        df = miRNA_scaled_df

    elif omics_name == "CN":
        absolute_copy_number = pd.read_csv(f'{data_directory}{file_names[7]}')
        absolute_copy_number = absolute_copy_number.rename(columns={absolute_copy_number.columns[0]: "CellLineID"})
        absolute_copy_number.columns = ['CellLineID'] + [f"7_{col}" for col in absolute_copy_number.columns[1:]]

        CN_cnv = absolute_copy_number.iloc[:, 1:]
        pca_CN = PCA(n_components=500)
        CN_pca = pca_CN.fit_transform(CN_cnv)
        CN_pca_df = pd.DataFrame(CN_pca)
        CN_pca_df.insert(0, "CellLineID", absolute_copy_number["CellLineID"].values)
        CN_pca_df.columns = ['CellLineID'] + [f"7_{col}" for col in CN_pca_df.columns[1:]]
        df = CN_pca_df
    else:
        print(f"{omics_name} is Not a correct omics data name")
        return
    
    return df

def get_dataloader_for_each(omics_name): # for individual models
    omics = get_omics_data(omics_name)
    cls = get_CLS_IC50_AUC()
    df_merged = pd.merge(cls, omics, left_on='CellLineID', right_on='CellLineID', how='inner')
    X = df_merged.drop(columns=["CellLineID", "DrugNumber",	"smiles", "AUC", "IC50"])
    y = df_merged[["AUC", "IC50"]]
    cls_array = X["CLS"]
    omics_array = X.drop(columns="CLS").to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)

    cls_tensor = torch.tensor(cls_array)#.to_numpy(dtype=np.float32))
    omics_tensor = torch.tensor(omics_array)
    target_tensor = torch.tensor(y_array)
    train_smiles, test_smiles, train_omics, test_omics, train_y, test_y = train_test_split(
        cls_tensor, omics_tensor, target_tensor, test_size=0.1, random_state=42
    )

    train_smiles, val_smiles, train_omics, val_omics, train_y, val_y = train_test_split(
        train_smiles, train_omics, train_y, test_size=0.1, random_state=42
    )
    train_dataset = TensorDataset(train_smiles, train_omics, train_y)
    val_dataset = TensorDataset(val_smiles, val_omics, val_y)
    test_dataset = TensorDataset(test_smiles, test_omics, test_y)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def get_dataloader_for_all(): # for final prediction model
    # call get_omics_data for all individual files
    expression = get_omics_data("expression")
    dmg_mut = get_omics_data("damaging_mutation")
    protein = get_omics_data("protein")
    hot_mut = get_omics_data("hotspot_mutation")
    metabolomics = get_omics_data("metabolomics")
    rna = get_omics_data("miRNA")
    cn = get_omics_data("CN")
    cls = get_CLS_IC50_AUC()
    cls = cls.drop(columns=["DrugNumber", "smiles"])

    # merge with CLS_IC50_AUC
    expression_n = expression.columns.tolist()
    dmg_mut_n = dmg_mut.columns.tolist()
    protein_n = protein.columns.tolist()
    hot_mut_n = hot_mut.columns.tolist()
    meta_n = metabolomics.columns.tolist()
    RNA_n = rna.columns.tolist()
    CN_n = cn.columns.tolist()
    IC50_AUC_CLS_n = cls.columns.tolist()

    all_data = dmg_mut.merge(expression, on='CellLineID', how = 'inner')
    all_data = all_data.merge(protein, on='CellLineID', how = 'inner')
    all_data = all_data.merge(hot_mut, on='CellLineID', how = 'inner')
    all_data = all_data.merge(metabolomics, on='CellLineID', how = 'inner')
    all_data = all_data.merge(rna, on='CellLineID', how = 'inner')
    all_data = all_data.merge(cn, on='CellLineID', how = 'inner')
    all_data = all_data.merge(cls, on='CellLineID', how = 'inner')

    expression = all_data[expression_n[1:]]
    damaging_mutation = all_data[dmg_mut_n[1:]]
    protein = all_data[protein_n[1:]]
    hotspot_mutation = all_data[hot_mut_n[1:]]
    metabolomic = all_data[meta_n[1:]]
    miRNA = all_data[RNA_n[1:]]
    copy_number = all_data[CN_n[1:]]
    target = all_data[["AUC", "IC50"]]
    cls = all_data["CLS"]

    # convert to tensor
    expression_tensor = torch.tensor(expression.to_numpy(dtype=np.float32))
    damaging_mutation_tensor = torch.tensor(damaging_mutation.to_numpy(dtype=np.float32))
    protein_tensor = torch.tensor(protein.to_numpy(dtype=np.float32))
    hotspot_mutation_tensor = torch.tensor(hotspot_mutation.to_numpy(dtype=np.float32))
    metabolomic_tensor = torch.tensor(metabolomic.to_numpy(dtype=np.float32))
    miRNA_tensor = torch.tensor(miRNA.to_numpy(dtype=np.float32))
    copy_number_tensor = torch.tensor(copy_number.to_numpy(dtype=np.float32))
    target_tensor = torch.tensor(target.to_numpy(dtype=np.float32))
    cls_tensor = torch.tensor(cls)

    # train, validation, test split
    train_smiles, test_smiles, train_expression, test_expression, train_damaging_mutation, test_damaging_mutation, train_protein, test_protein, train_hotspot_mutation, test_hotspot_mutation, train_metabolomic, test_metabolomic, train_miRNA, test_miRNA, train_copy_number, test_copy_number, train_y, test_y = train_test_split(
        cls_tensor, expression_tensor, damaging_mutation_tensor, protein_tensor, hotspot_mutation_tensor, metabolomic_tensor, miRNA_tensor, copy_number_tensor, target_tensor, test_size=0.1, random_state=42)

    train_smiles, val_smiles, train_expression, val_expression, train_damaging_mutation, val_damaging_mutation, train_protein, val_protein, train_hotspot_mutation, val_hotspot_mutation, train_metabolomic, val_metabolomic, train_miRNA, val_miRNA, train_copy_number, val_copy_number, train_y, val_y = train_test_split(
        train_smiles, train_expression, train_damaging_mutation, train_protein, train_hotspot_mutation, train_metabolomic, train_miRNA, train_copy_number, train_y, test_size=0.1, random_state=42)

    train_dataset = TensorDataset(train_smiles, train_expression, train_damaging_mutation, train_protein, train_hotspot_mutation, train_metabolomic, train_miRNA, train_copy_number, train_y)
    val_dataset = TensorDataset(val_smiles, val_expression, val_damaging_mutation, val_protein, val_hotspot_mutation, val_metabolomic, val_miRNA, val_copy_number, val_y)
    test_dataset = TensorDataset(test_smiles, test_expression, test_damaging_mutation, test_protein, test_hotspot_mutation, test_metabolomic, test_miRNA, test_copy_number, test_y)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def train_individual_model(folder_path, omics_name, train_loader, val_loader, num_epochs, patience):
    omics_num = None
    if omics_name == "expression":
        omics_num = 1
    elif omics_name == "damaging_mutation":
        omics_num = 2
    elif omics_name == "protein":
        omics_num = 3
    elif omics_name == "hotspot_mutation":
        omics_num = 4
    elif omics_name == "metabolomics":
        omics_num = 5
    elif omics_name == "miRNA":
        omics_num = 6
    elif omics_name == "CN":
        omics_num = 7
    else:
        print(f"Wrong omics name : {omics_name}")
        return

    #train_loader, val_loader, test_loader = get_dataloader_for_each(omics_name)
    cls, features, labels = next(iter(train_loader))
    num_omics = features.shape[1]
    cls_dim = cls.shape[1]

    # hyperparamter
    d_model = 512 # can be adjust, should be divisible by nhead
    nhead = 8 # can be adjust
    num_layers = 4 # can be adjust

    # model setting
    model = MultiTaskRegressionTransformer(cls_dim, d_model, nhead, num_layers, num_omics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    #num_epochs = 1000
    #patience = 100 # for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_model_filename = None

    # training loop
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0.0
        for batch in train_loader:
            smiles_list, omics_data, targets = batch
            ic50_targets, auc_targets = targets[:, 0], targets[:, 1]
            
            smiles_list = smiles_list.to(device)
            omics_data = omics_data.to(device)
            ic50_targets = ic50_targets.to(device)
            auc_targets = auc_targets.to(device)
            
            optimizer.zero_grad()
            ic50_pred, auc_pred = model(smiles_list, omics_data)

            # weight for auc loss and IC50 loss
            w_ic50 = 2.0 # AUC is trained easier than IC50, so set IC50 weight larger (need further experiment)
            w_auc = 1.0   
            loss = 0.0
            
            # compute loss for valid target (which is not NAN)
            ic50_mask = ~torch.isnan(ic50_targets)
            auc_mask = ~torch.isnan(auc_targets)
            
            if ic50_mask.sum() > 0:
                loss_ic50 = criterion(ic50_pred[ic50_mask], ic50_targets[ic50_mask].unsqueeze(1))
                loss += w_ic50 * loss_ic50
            if auc_mask.sum() > 0:
                loss_auc = criterion(auc_pred[auc_mask], auc_targets[auc_mask].unsqueeze(1))
                loss += w_auc * loss_auc
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                smiles_list, omics_data, targets = batch
                ic50_targets, auc_targets = targets[:, 0], targets[:, 1]
                
                smiles_list = smiles_list.to(device)
                omics_data = omics_data.to(device)
                ic50_targets = ic50_targets.to(device)
                auc_targets = auc_targets.to(device)
                
                ic50_pred, auc_pred = model(smiles_list, omics_data)
                
                loss = 0.0
                ic50_mask = ~torch.isnan(ic50_targets)
                auc_mask = ~torch.isnan(auc_targets)
                
                if ic50_mask.sum() > 0:
                    loss_ic50 = criterion(ic50_pred[ic50_mask], ic50_targets[ic50_mask].unsqueeze(1))
                    loss += loss_ic50
                if auc_mask.sum() > 0:
                    loss_auc = criterion(auc_pred[auc_mask], auc_targets[auc_mask].unsqueeze(1))
                    loss += loss_auc
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
        
        # best model saving & early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() 
            best_model_filename = f"{folder_path}/{omics_num}_best_model_{omics_name}_{str(best_val_loss)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            print("Validation loss improved. Best model updated.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # save the best model
    if best_model_state is not None:
        torch.save(best_model_state, best_model_filename)
        print(f"Best model saved to {best_model_filename}")
     
    return best_model_filename

def train_final_model(folder_path, train_loader, val_loader, num_epochs, patience):
    cls, feature1, feature2, feature3, feature4, feature5, feature6, feature7, labels = next(iter(train_loader))
    num_omics = [feature1.shape[1], feature2.shape[1], feature3.shape[1], feature4.shape[1], feature5.shape[1], feature6.shape[1], feature7.shape[1]]
    cls_dim = cls.shape[1]

    # hyperparamter
    d_model = 512 # can be adjust, should be divisible by nhead
    nhead = 8 # can be adjust
    num_layers = 4 # can be adjust

    model_names = os.listdir(folder_path)

    model = FinalModel(folder_path, model_names, cls_dim, d_model, nhead, num_layers, num_omics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_model_filename = None

    # training loop
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0.0
        for batch in train_loader:
            smiles_list, expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data, targets = batch
            omics_data = [expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data]
            ic50_targets, auc_targets = targets[:, 0], targets[:, 1]
            
            smiles_list = smiles_list.to(device)
            omics_data = [data.to(device) for data in omics_data]
            ic50_targets = ic50_targets.to(device)
            auc_targets = auc_targets.to(device)
            
            optimizer.zero_grad()
            ic50_pred, auc_pred = model(smiles_list, omics_data)

            # weight for auc loss and IC50 loss
            w_ic50 = 2.0 # AUC is trained easier than IC50, so set IC50 weight larger (need further experiment)
            w_auc = 1.0   
            loss = 0.0
            
            # compute loss for valid target (which is not NAN)
            ic50_mask = ~torch.isnan(ic50_targets)
            auc_mask = ~torch.isnan(auc_targets)
            
            if ic50_mask.sum() > 0:
                loss_ic50 = criterion(ic50_pred[ic50_mask], ic50_targets[ic50_mask])
                loss += w_ic50 * loss_ic50
            if auc_mask.sum() > 0:
                loss_auc = criterion(auc_pred[auc_mask], auc_targets[auc_mask])
                loss += w_auc * loss_auc
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                smiles_list, expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data, targets = batch
                omics_data = [expression_data, damaging_mutation_data, protein_data, hotspot_mutation_data, metabolomic_data, miRNA_data, copy_number_data]
                ic50_targets, auc_targets = targets[:, 0], targets[:, 1]
                
                smiles_list = smiles_list.to(device)
                omics_data = [data.to(device) for data in omics_data]
                ic50_targets = ic50_targets.to(device)
                auc_targets = auc_targets.to(device)
                
                ic50_pred, auc_pred = model(smiles_list, omics_data)
                
                loss = 0.0
                ic50_mask = ~torch.isnan(ic50_targets)
                auc_mask = ~torch.isnan(auc_targets)
                
                if ic50_mask.sum() > 0:
                    loss_ic50 = criterion(ic50_pred[ic50_mask], ic50_targets[ic50_mask])
                    loss += loss_ic50
                if auc_mask.sum() > 0:
                    loss_auc = criterion(auc_pred[auc_mask], auc_targets[auc_mask])
                    loss += loss_auc
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
        
        # best model saving & early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() 
            best_model_filename = f"../best_models/best_final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            print("Validation loss improved. Best model updated.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # save the best model
    if best_model_state is not None:
        torch.save(best_model_state, best_model_filename)
        print(f"Best model saved to {best_model_filename}")
    
    return best_model_filename

def test_model(omics_name, model_filename, test_loader):
    cls, features, labels = next(iter(test_loader))
    num_omics = features.shape[1]
    cls_dim = cls.shape[1]

    # hyperparamter
    d_model = 512 # can be adjust, should be divisible by nhead
    nhead = 8 # can be adjust
    num_layers = 4 # can be adjust

    # model setting
    model = MultiTaskRegressionTransformer(cls_dim, d_model, nhead, num_layers, num_omics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()

    ic50_preds, ic50_targets = [], []
    auc_preds, auc_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            smiles_list, omics_data, targets = batch
            ic50_true, auc_true = targets[:, 0], targets[:, 1]

            smiles_list = smiles_list.to(device)
            omics_data = omics_data.to(device)
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
        if len(auc_preds) > 0:
            metrics["AUC_MSE"] = mean_squared_error(auc_targets, auc_preds)
            metrics["AUC_MAE"] = mean_absolute_error(auc_targets, auc_preds)
            metrics["AUC_R2"] = r2_score(auc_targets, auc_preds)

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
            plt.title(f"IC50 Scatter Plot ({omics_name})")
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
            plt.title(f"AUC Scatter Plot ({omics_name})")
            plt.text(0.05, 0.90, f"R2 = {metrics['AUC_R2']:.4f}", transform=plt.gca().transAxes, 
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            plt.legend(loc="lower right")
            plt.show()
        
        return metrics
    
def test_final_model(folder_path, model_filename, test_loader):
    #folder_path = '../final_models'  
    model_names = os.listdir(folder_path)

    #'model_paths', 'cls_dim', 'd_model', 'nhead', 'num_layers', and 'omics_dimension_list'
    cls, feature1, feature2, feature3, feature4, feature5, feature6, feature7, labels = next(iter(test_loader))
    omics_dim_list = [feature1.shape[1], feature2.shape[1], feature3.shape[1], feature4.shape[1], feature5.shape[1], feature6.shape[1], feature7.shape[1]]
    cls_dim = cls.shape[1]

    # hyperparamter
    d_model = 512 # can be adjust, should be divisible by nhead
    nhead = 8 # can be adjust
    num_layers = 4 # can be adjust
    
    model = FinalModel(folder_path, model_names, cls_dim, d_model, nhead, num_layers, omics_dim_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(model_filename, map_location=device))

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
        if len(auc_preds) > 0:
            metrics["AUC_MSE"] = mean_squared_error(auc_targets, auc_preds)
            metrics["AUC_MAE"] = mean_absolute_error(auc_targets, auc_preds)
            metrics["AUC_R2"] = r2_score(auc_targets, auc_preds)

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
            plt.title("AUC Scatter Plot")
            plt.text(0.05, 0.90, f"R2 = {metrics['AUC_R2']:.4f}", transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            plt.legend(loc="lower right")
            plt.show()
        
        return metrics