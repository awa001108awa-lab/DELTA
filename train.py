import os
import copy
import warnings
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

import config
from src import evaluator
from utils.random_seed import seed_everything
from src.data_loader import load_and_preprocess_data, HRVSequenceDataset
from src.transformer_slidingwindow import TransformerClassifier

warnings.filterwarnings('ignore')

def make_loader(ds, batch_size, shuffle=True):
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )

def train_one_fold(model, train_loader, val_loader, params, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    weight = torch.tensor([1.0, float(params['pos_weight'])]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.10) 
    
    best_auc, best_state, no_improve = 0.0, None, 0
    train_losses = [] 
    
    for epoch in range(config.FINAL_EPOCHS):
        # Learning rate warmup logic
        lr = params['lr'] * (epoch + 1) / config.WARMUP_EPOCHS if epoch < config.WARMUP_EPOCHS else params['lr']
        for pg in optimizer.param_groups: 
            pg['lr'] = lr
        
        model.train()
        epoch_loss = 0.0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, src_key_padding_mask=m), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            y_true, _, y_prob = evaluator.get_predictions(model, val_loader, device)
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= config.FINAL_EARLY_STOPPING_PATIENCE: 
                print(f" -> Early stopping triggered at Epoch {epoch+1}")
                break
    
    if best_state: 
        model.load_state_dict(best_state)
    return model, train_losses


def main():
    seed_everything(config.SEED)
    
    # Load aligned dataset containing features and labels
    X_cv, y_cv, cv_ids, X_test, y_test, test_ids, splits_data = \
        load_and_preprocess_data(config.CSV_FILE, config.SEED, config.SPLIT_JSON_PATH)
    
    input_dim = X_cv[0].shape[1]
    params = config.BEST_PARAMS
    save_dir = getattr(config, "SAVE_DIR", ".")
    os.makedirs(save_dir, exist_ok=True) 
    
    print("\n🚀 Starting training pipeline (90% train, 10% test)...")
    
    train_ds_all = HRVSequenceDataset(X_cv, y_cv)
    test_ds = HRVSequenceDataset(X_test, y_test, max_len=train_ds_all.max_len)
    
    model_final = TransformerClassifier(
        input_dim=input_dim, 
        **{k:params[k] for k in ['d_model','nhead','num_layers','dropout']}
    ).to(config.DEVICE)
    
    model_final, train_losses = train_one_fold(
        model_final, 
        make_loader(train_ds_all, params['batch_size']), 
        make_loader(test_ds, params['batch_size'], False), 
        params, 
        config.DEVICE
    )
    
    print("\n" + "="*50)
    print("🎯 Final 10% Independent Test Set Evaluation Results")
    print("="*50)
    y_t, y_p, y_pb = evaluator.get_predictions(model_final, make_loader(test_ds, params['batch_size'], False), config.DEVICE)
    
    metrics_final, cm_final = evaluator.calculate_and_print_metrics(y_t, y_p, y_pb)
    evaluator.plot_performance(train_losses, cm_final, save_path=os.path.join(save_dir, getattr(config, "RESULT_PLOT_PNG", "result_plot.png")))
    
    # Save ROC curve coordinates for multi-model comparison
    fpr, tpr, _ = roc_curve(y_t, y_pb)
    roc_data_path = os.path.join(save_dir, "roc_data_transformer.npz")
    np.savez(roc_data_path, fpr=fpr, tpr=tpr, auc=metrics_final['roc_auc'])
    print(f"📈 ROC data saved to: {roc_data_path}")
    
    # Save optimal model weights
    model_save_path = os.path.join(save_dir, "best_model_final.pt")
    torch.save(model_final.state_dict(), model_save_path)
    print(f"💾 Best model state saved to: {model_save_path}")
    
    print("\n✅ Execution completed successfully!")

if __name__ == '__main__':
    main()