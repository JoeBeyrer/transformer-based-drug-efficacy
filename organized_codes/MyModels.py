import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# individual model
class MultiTaskRegressionTransformer(nn.Module):
    def __init__(self, chemberta_dim, d_model, nhead, num_layers, num_omics):
        super(MultiTaskRegressionTransformer, self).__init__()
        
        # CLS expression for SMILES is already embedded seperately by ChemBERTa
        self.chemberta_dim = chemberta_dim

        # Omics -> Transformer Encoder
        self.omics_fc = nn.Linear(num_omics, d_model)
        self.omics_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.omics_transformer = nn.TransformerEncoder(self.omics_encoder_layer, num_layers=num_layers)
        
        # cls -> embed to d_model dimension
        self.cls_proj = nn.Linear(self.chemberta_dim, d_model)
        
        # Cross-Attention (omics and drug data were embedded by different encoder, so we need to train their relationship additionally)
        # SMILES -> omics
        self.cross_attn_smiles = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        # Omics -> SMILES
        self.cross_attn_omics = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # fuse two cross-attn information
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # prediction heads
        self.ic50_head = nn.Linear(d_model, 1)
        self.auc_head = nn.Linear(d_model, 1)
        
    def forward(self, cls_data, omics_data):
        """
        cls_data: SMILES CLS expression (batch_size, chemberta_dim)
        omics_data: (batch_size, num_omics) 
        """
        
        # Omics: fc → Transformer Encoder
        omics_repr = self.omics_fc(omics_data) # (batch_size, d_model)
        omics_repr = omics_repr.unsqueeze(1) # (batch_size, 1, d_model)
        omics_repr = self.omics_transformer(omics_repr) # (batch_size, 1, d_model)
        omics_repr = omics_repr.squeeze(1) # (batch_size, d_model)
        
        # project SMILES to d_model
        smiles_proj = self.cls_proj(cls_data) # (batch_size, d_model)
        
        # (batch_first=True → [batch, seq_len, d_model])
        smiles_seq = smiles_proj.unsqueeze(1) # (batch_size, 1, d_model)
        omics_seq = omics_repr.unsqueeze(1) # (batch_size, 1, d_model)
        
        # Cross-Attention
        # 1. SMILES -> omics: query=smiles, key/value=omics
        attn_smiles, _ = self.cross_attn_smiles(query=smiles_seq, key=omics_seq, value=omics_seq)
        # 2. Omics -> SMILES: query=omics, key/value=smiles
        attn_omics, _ = self.cross_attn_omics(query=omics_seq, key=smiles_seq, value=smiles_seq)
        
        # Residual connection (add attention result to original data) -> reserve original data
        smiles_updated = smiles_proj + attn_smiles.squeeze(1) # (batch_size, d_model)
        omics_updated = omics_repr + attn_omics.squeeze(1) # (batch_size, d_model)
        
        # Send to MLP
        fused = torch.cat([smiles_updated, omics_updated], dim=1) # (batch_size, 2*d_model)
        fused = self.fusion_mlp(fused) # (batch_size, d_model)
        
        # Final prediction
        ic50_pred = self.ic50_head(fused) # (batch_size, 1)
        auc_pred = self.auc_head(fused) # (batch_size, 1)
        
        return ic50_pred, auc_pred
    

class FinalModel(nn.Module):
    def __init__(self, model_folder, model_paths, cls_dim, d_model, nhead, num_layers, omics_dimension_list):
        super(FinalModel, self).__init__()
        
        # call pretrained models, and freeze
        self.pretrained_models = nn.ModuleList()
        for i in range(7):
            model = MultiTaskRegressionTransformer(cls_dim, d_model, nhead, num_layers, omics_dimension_list[i])
            model_path = model_folder + "/" + model_paths[i]
            model.load_state_dict(torch.load(model_path))
    
            for param in model.parameters():
                param.requires_grad = False
            self.pretrained_models.append(model)
        
        self.fc1 = nn.Linear(14, 64) # 2 * 7
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2) 

    def forward(self, drug_input, omics_inputs):
        """
        drug_input: Tensor, shape (batch_size, drug_input_dim)
        omics_inputs: list of 7 Tensors
        """
        ic50_preds = []
        auc_preds = []

        # (batch_size,)
        for i, model in enumerate(self.pretrained_models):
            ic50_pred, auc_pred = model(drug_input, omics_inputs[i]) # (batch_size,), (batch_size,)
            
            # (batch_size, 1)
            ic50_preds.append(ic50_pred.unsqueeze(1)) # I guess unsqueeze is unnecessary
            auc_preds.append(auc_pred.unsqueeze(1))
            #print(ic50_preds.shape)

        ic50_preds = torch.cat(ic50_preds, dim=1) # (batch_size, 7, 1)
        auc_preds = torch.cat(auc_preds, dim=1)   # (batch_size, 7, 1)
        #print(ic50_preds.shape)

        concat_preds = torch.cat([ic50_preds, auc_preds], dim=1) # (batch_size, 14, 1)
        concat_preds = concat_preds.squeeze(2)
        #print(concat_preds.shape)
        
        x = F.relu(self.fc1(concat_preds))
        x = F.relu(self.fc2(x))
        output = self.fc3(x) # (batch_size, 2)

        return output[:, 0], output[:, 1] # (batch_size,), (batch_size,)



