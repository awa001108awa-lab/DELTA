import torch

# === Data and Path Configurations ===
CSV_FILE = "processed_features/Processed_HRV_Lead2.csv"
SAVE_DIR = "./results"
SPLIT_JSON_PATH = "dataset_splits.json"

# === Optimal Hyperparameters ===
BEST_PARAMS = {
    'd_model': 128,
    'num_layers': 2,
    'nhead': 8,
    'dropout': 0.1,
    'lr': 0.00005,
    'batch_size': 128,
    'pos_weight': 3.0
}

# === Training Controls ===
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_EPOCHS = 100
WARMUP_EPOCHS = 5
FINAL_EARLY_STOPPING_PATIENCE = 10

# === Hardware Acceleration ===
NUM_WORKERS = 2
PIN_MEMORY = True