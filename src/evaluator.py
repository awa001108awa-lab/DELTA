import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    roc_auc_score, confusion_matrix, matthews_corrcoef, 
    cohen_kappa_score, balanced_accuracy_score, 
    average_precision_score
)

def get_predictions(model, loader, device):
    model.eval()
    all_targets = []  
    all_probs = []  
    all_preds = []  

    with torch.no_grad():
        for inputs, targets, masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            logits = model(inputs, src_key_padding_mask=masks)
            probs = torch.softmax(logits, dim=1)[:, 1]  
            preds = (probs > 0.5).long()

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def calculate_and_print_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        else:
            roc_auc = 0.5
            pr_auc = 0.0
    except:
        roc_auc = 0.5
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    else:
        specificity = 0.0
        npv = 0.0

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0

    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except:
        kappa = 0.0

    try:
        bacc = balanced_accuracy_score(y_true, y_pred)
    except:
        bacc = 0.0

    metrics = {
        'accuracy': float(acc),
        'recall': float(recall),
        'specificity': float(specificity),
        'precision': float(precision),
        'npv': float(npv),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'mcc': float(mcc),
        'kappa': float(kappa),
        'bacc': float(bacc),
        'confusion_matrix': cm.tolist()
    }

    print("-" * 50)
    print("📊 Final Evaluation Results:")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"Specificity   : {specificity:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"NPV           : {npv:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print(f"PR-AUC        : {pr_auc:.4f}")
    print(f"MCC           : {mcc:.4f}")
    print(f"Kappa         : {kappa:.4f}")
    print(f"BACC          : {bacc:.4f}")
    print("-" * 50)

    return metrics, cm

def plot_performance(train_losses, cm, save_path=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if train_losses and isinstance(train_losses[0], (list, np.ndarray)):
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(train_losses), 1)))
        for i, f in enumerate(train_losses):
            arr = np.asarray(f, dtype=float)
            x = np.arange(0, len(arr)) 
            plt.plot(x, arr, label=f'Fold {i + 1}', color=colors[i % 10], linewidth=1.5, alpha=0.9)
        plt.legend()
    else:
        plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.legend()
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📈 Plot saved to: {save_path}")
        
    plt.close()

def save_metrics(metrics, save_path="result_metrics.json"):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"📝 Metrics saved to: {save_path}")

def save_metrics_txt(metrics, save_path="result_metrics.txt"):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("📊 Final Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy      : {metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Recall        : {metrics.get('recall', 0):.4f}\n")
        f.write(f"Specificity   : {metrics.get('specificity', 0):.4f}\n")
        f.write(f"Precision     : {metrics.get('precision', 0):.4f}\n")
        f.write(f"NPV           : {metrics.get('npv', 0):.4f}\n")
        f.write(f"F1 Score      : {metrics.get('f1_score', 0):.4f}\n")
        f.write(f"ROC-AUC       : {metrics.get('roc_auc', metrics.get('auc', 0)):.4f}\n")
        f.write(f"PR-AUC        : {metrics.get('pr_auc', 0):.4f}\n")
        f.write(f"MCC           : {metrics.get('mcc', 0):.4f}\n")
        f.write(f"Kappa         : {metrics.get('kappa', 0):.4f}\n")
        f.write(f"BACC          : {metrics.get('bacc', 0):.4f}\n")
        f.write("-" * 50 + "\n")
        f.write("Confusion Matrix:\n")
        cm = metrics['confusion_matrix']
        f.write(f"        Pred 0  Pred 1\n")
        f.write(f"True 0   {cm[0][0]:4d}  {cm[0][1]:4d}\n")
        f.write(f"True 1   {cm[1][0]:4d}  {cm[1][1]:4d}\n")
        f.write("=" * 50 + "\n")
    print(f"📝 Metrics saved to: {save_path}")