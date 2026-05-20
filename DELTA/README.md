# DELTA: Transformer-based Prediction of Incident MASLD using Exercise ECG HRV Features

This repository contains the official PyTorch implementation for the DELTA model, a Transformer-based deep learning framework designed to predict incident Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD) using sliding-window Heart Rate Variability (HRV) features extracted from exercise electrocardiogram (ECG) data.

# Repository Structure

```text
.
├── data_preprocess.py       # ECG cleaning & sliding-window HRV feature extraction
├── config.py                   # Global hyperparameters & path configurations
├── train.py                    # Main training and evaluation pipeline
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_loader.py          # Data alignment, imputation, and Dataset generation
│   ├── transformer_slidingwindow.py # DELTA Transformer architecture
│   └── evaluator.py            # Metrics calculation and visualization tools
└── utils/
    └── random_seed.py          # Deterministic seed settings

# Requirements
This code was tested on Python 3.12.12 with pytorch '2.3.1 '.

## Data Preparation (Due to Privacy Restrictions)

Due to the strict data usage agreements of the UK Biobank and patient privacy regulations, the raw ECG XML files and clinical labels cannot be shared publicly. To reproduce the pipeline or train on your own cohort, please structure your local data/ directory as follows:

Plaintext

data/
├── id.csv                  # CSV containing patient IDs (Column 1) and Binary Labels (Column 2)
├── HRV_indices.csv         # Target HRV indices to extract (Column 2)
└── ecg_raw/                # Directory containing raw multi-lead ECG XML files
    ├── control
    └── MASLD/
Note: The preprocessing script (data_preprocess.py) is designed to automatically extract the target ECG lead, calculate sliding-window HRV indices matching HRV_indices.csv, and handle missing values.

# How to Run
Step 1: Feature Extraction
Run the preprocessing script to clean the signals, extract R-peaks, and compute the sliding-window HRV features.

python data_preprocess.py
This will generate Processed_HRV_Lead2.csv in the output directory.

Step 2: Train the DELTA Model
Once the feature extraction is complete, you can train the model. Hyperparameters can be adjusted directly in config.py.

python train.py

# Outputs
Upon completion, the script will automatically generate and save the following in the ./results/ folder:

result_plot.png: Training loss curve and testing confusion matrix.

roc_data_transformer.npz: Coordinates for plotting the ROC curve.

best_model_final.pt: The optimal model weights.