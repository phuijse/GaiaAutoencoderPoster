import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from umap.umap_ import fuzzy_simplicial_set

#https://github.com/lmcinnes/umap/tree/master?tab=BSD-3-Clause-1-ov-file#readme

def get_graph_elements(graph_, n_epochs):
    graph = graph_.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    epochs_per_sample = n_epochs * graph.data
    head = graph.row
    tail = graph.col
    weight = graph.data
    return graph, epochs_per_sample, head, tail, weight, n_vertices


# Adapted from https://github.com/lmcinnes/umap/blob/cfbf23ee1787eaf73c4f13c5d76315b612f55b03/umap/parametric_umap.py#L826
class UMAPDataset(Dataset):
    def __init__(self, data, n_neighbors=15, metric='euclidean', n_epochs=200, random_state=1234):
        self.data = torch.Tensor(data.astype('float32'))
        graph, _, _ = fuzzy_simplicial_set(
            data,
            n_neighbors = n_neighbors,
            metric = metric,
            random_state = random_state,
        )
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(graph, n_epochs)
        
        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        
        
    def __len__(self):
        return int(self.data.shape[0])
    
    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)


#Adapted from https://github.com/lmcinnes/umap/blob/cfbf23ee1787eaf73c4f13c5d76315b612f55b03/umap/parametric_umap.py#L715
def compute_cross_entropy(probabilities_graph, log_probabilities_distance, eps=1e-4, repulsion_strength=1.0):
    attraction_term = - probabilities_graph * nn.functional.logsigmoid(log_probabilities_distance)
    repellant_term = -(1.0 - probabilities_graph) * (
        nn.functional.logsigmoid(log_probabilities_distance) - log_probabilities_distance
    ) * repulsion_strength
    cross_entropy = attraction_term + repellant_term
    return attraction_term, repellant_term, cross_entropy


# Adapted https://github.com/lmcinnes/umap/blob/cfbf23ee1787eaf73c4f13c5d76315b612f55b03/umap/parametric_umap.py#L1190
def umap_loss(embedding_to, embedding_from, batch_size, negative_sample_rate=5):
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat((
        (embedding_to - embedding_from).norm(dim=1),
        (embedding_neg_to - embedding_neg_from).norm(dim=1)
    ), dim=0)
    
    log_probabilities_distance = - torch.log1p(distance_embedding.pow(2))
    
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size, device=log_probabilities_distance.device), 
         torch.zeros(batch_size * negative_sample_rate, device=log_probabilities_distance.device)
        ), dim=0
    )
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph,
        log_probabilities_distance,
    )
    return ce_loss.mean()
