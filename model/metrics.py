import torch
from tqdm import tqdm
from collections import defaultdict


def calculate_sim_mat(embeddings: torch.Tensor):
    normal_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    
    sim_mat = normal_embeddings@normal_embeddings.T
    sim_mat.fill_diagonal_(-1.0)  # exclude self similarity as it's always 1.0
    return sim_mat


def calculate_map_at_R(embeddings: torch.Tensor, labels, R=499):
    assert embeddings.shape[0] == len(labels), "Number of embeddings and labels must match."
    
    print(f"Calculating MAP@{R} score.")
    
    N = len(labels)
    label_counts = defaultdict(lambda:-1)
    for label in labels: label_counts[label.item()] += 1
    
    map_score = 0.0
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
        map_score += ap
    # Normalize by number of queries to get MAP
    map_score /= N
    return map_score
