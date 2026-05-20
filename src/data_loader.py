import pandas as pd
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.impute import SimpleImputer

class HRVSequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_len=None):
        self.labels = torch.tensor(labels, dtype=torch.long)
        curr_max = max([len(s) for s in sequences]) if len(sequences) > 0 else 0
        self.max_len = max_len if max_len is not None else curr_max
        if len(sequences) > 0:
            self.feature_dim = sequences[0].shape[1]
            self.padded_sequences = torch.zeros(len(sequences), self.max_len, self.feature_dim)
            self.masks = torch.ones(len(sequences), self.max_len, dtype=torch.bool)
            for i, seq in enumerate(sequences):
                seq_len = min(len(seq), self.max_len)
                self.padded_sequences[i, :seq_len, :] = torch.tensor(np.array(seq[:seq_len], dtype=np.float32), dtype=torch.float32)
                self.masks[i, :seq_len] = False
        else:
            self.feature_dim = 0
            self.padded_sequences = torch.empty(0)
            self.masks = torch.empty(0)

    def __len__(self): 
        return len(self.labels)
        
    def __getitem__(self, idx): 
        return self.padded_sequences[idx], self.labels[idx], self.masks[idx]

def df_to_sequences(df, feature_cols):
    sequences, labels, sample_ids = [], [], []
    if 'Window_Index' in df.columns:
        df = df.sort_values(by=['Filename', 'Window_Index'])
    for filename, group in df.groupby('Filename'):
        sequences.append(group[feature_cols].values)
        labels.append(group['Group'].iloc[0])
        sample_ids.append(filename)
    return sequences, labels, sample_ids

def load_and_preprocess_data(csv_file, seed, split_json_path="dataset_splits.json"):
    df = pd.read_csv(csv_file)
    
    non_feature_cols = ['Filename', 'Group', 'Lead_Used', 'Window_Index', 'Window_Start_Time_s', 'Window_End_Time_s']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='mean')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    unique_patients = df['Filename'].unique()
    patient_labels = df.groupby('Filename')['Group'].first().loc[unique_patients].values

    if os.path.exists(split_json_path):
        print(f"Loading existing ID splits from {split_json_path}...")
        with open(split_json_path, 'r', encoding='utf-8') as f:
            splits_data = json.load(f)
        train_val_patients = splits_data['cv_pool_patients']
        test_patients = splits_data['test_patients']
    else:
        print(f"First run: Generating ID splits and saving to {split_json_path}...")
        train_val_patients, test_patients = train_test_split(
            unique_patients, test_size=0.1, stratify=patient_labels, random_state=seed
        )
        train_val_patients, test_patients = train_val_patients.tolist(), test_patients.tolist()

        # 模拟生成五折划分 ID
        cv_df_tmp = df[df['Filename'].isin(train_val_patients)].copy()
        tmp_ids, tmp_labels = [], []
        for filename, group in cv_df_tmp.groupby('Filename'):
            tmp_ids.append(filename)
            tmp_labels.append(group['Group'].iloc[0])
            
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = []
        for fold, (t_idx, v_idx) in enumerate(sgkf.split(np.zeros(len(tmp_ids)), tmp_labels, groups=tmp_ids)):
            folds.append({
                "fold": fold + 1,
                "train_patients": np.array(tmp_ids)[t_idx].tolist(),
                "val_patients": np.array(tmp_ids)[v_idx].tolist()
            })
        splits_data = {"test_patients": test_patients, "cv_pool_patients": train_val_patients, "cv_folds": folds}
        with open(split_json_path, "w", encoding="utf-8") as f:
            json.dump(splits_data, f, indent=4)

    cv_df = df[df['Filename'].isin(train_val_patients)].copy()
    test_df = df[df['Filename'].isin(test_patients)].copy()

    scaler = StandardScaler()
    cv_df[feature_cols] = scaler.fit_transform(cv_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_cv, y_cv, cv_ids = df_to_sequences(cv_df, feature_cols)
    X_test, y_test, test_ids = df_to_sequences(test_df, feature_cols)

    return np.array(X_cv, dtype=object), np.array(y_cv), np.array(cv_ids), \
           np.array(X_test, dtype=object), np.array(y_test), np.array(test_ids), splits_data