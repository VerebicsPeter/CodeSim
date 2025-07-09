import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
)


def calculate_sim_mat(embeddings: torch.Tensor):
    normal_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    
    sim_mat = normal_embeddings@normal_embeddings.T
    sim_mat.fill_diagonal_(-1.0)  # exclude self similarity as it's always 1.0
    return sim_mat


def calculate_map_at_R(embeddings: torch.Tensor, labels, R=499):
    assert embeddings.shape[0] == len(labels), "Number of embeddings and labels must match."
    
    print(f"Calculating MAP@R={R} score.")
    
    N = len(labels)
    label_counts = defaultdict(lambda:-1)
    for label in labels: label_counts[label.item()] += 1
    
    map_r = 0.0
    sim_mat = calculate_sim_mat(embeddings)
    
    # Calculate MAP score of queries
    for i, true_label in tqdm(enumerate(labels), total=N):
        retrieved_indexes = sim_mat[i].topk(R).indices
        retrieved_labels  = [labels[j] for j in retrieved_indexes]  # labels per query
        # Calculate the average precision of results
        ap = 0.0  # AP per query
        nh = 0    # number of hits (true positives)
        for k, retrieved_label in enumerate(retrieved_labels):
            if retrieved_label == true_label:
                nh += 1
                ap += nh / (k+1)
        count = label_counts[true_label.item()]
        if nh > 0: ap /= count  # IMPORTANT: normalized by instance count not TP count
        map_r += ap
    # Normalize by number of queries to get MAP
    map_r /= N
    return {"map_r": map_r}


def calculate_map_metrics(all_embs, all_lbls):
    return calculate_map_at_R(all_embs, all_lbls, R=499)


def calculate_cls_metrics(y_true, y_pred):
    # NOTE: `y_pred` values are logits
    
    # F1 score (on best threshold)
    eps = 1e-8
    pre, rec, _ = precision_recall_curve(y_true, y_pred)
    f1s = 2 * (pre * rec) / (pre + rec + eps)
    f1 = f1s[np.argmax(f1s)]
    
    # ROC curve AUC
    roc_auc = roc_auc_score(y_true, y_pred)
    
    return {"F1": f1, "roc_auc": roc_auc}


def print_reports(y_true, y_pred, thresholds=(.5,.7,.9)):
    
    for threshold in thresholds:
        report = classification_report(y_true, [int(pred > threshold) for pred in y_pred])
        print(f"REPORT @ threshold={threshold}")
        print(report)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    _auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'(AUC = {_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")
    plt.show()
