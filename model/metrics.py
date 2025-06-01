import torch
import matplotlib.pyplot as plt


def calculate_sim_mat(embeddings: torch.Tensor, debug=False):
    normal_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    
    sim_mat = normal_embeddings@normal_embeddings.T
    sim_mat.fill_diagonal_(-1.0)  # exclude self similarity as it's always 1.0
    
    if debug:
        print("Sim matrix shape:", sim_mat.shape)
        plt.imshow(sim_mat.detach().cpu().numpy(), vmin=-1.0, vmax=1.0)
        plt.colorbar()
        plt.show()
    
    return sim_mat


def calculate_map_at_R(embeddings: torch.Tensor, labels, R=499):
    assert embeddings.shape[0] == len(labels), "Number of embeddings and labels must match."
    N = len(labels)  # number of queries
    map_score = 0.0
    sim_mat = calculate_sim_mat(embeddings)
    # Calculate MAP score of queries
    for i, true_label in enumerate(labels):
        retrieved_indexes = sim_mat[i].topk(R).indices
        retrieved_labels  = [labels[j] for j in retrieved_indexes]  # labels per query
        # Calculate the average precision of results
        ap = 0.0  # AP per query
        nh = 0    # number of hits (true positives)
        for k, retrieved_label in enumerate(retrieved_labels):
            if retrieved_label == true_label:
                nh += 1
                ap += nh / (k+1)
        # Normalize by number of hits (if there were any)
        if nh > 0: ap /= nh
        map_score += ap
    # Normalize by number of queries to get MAP
    map_score /= N
    return map_score
