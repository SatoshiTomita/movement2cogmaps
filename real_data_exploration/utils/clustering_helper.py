import numpy as np
from sklearn.metrics import silhouette_samples

def compute_silhouette_samples(distance_mat, clusters):
    return silhouette_samples(
        X=distance_mat,
        labels=clusters,
        metric='precomputed'
    )

def compute_explained_variance(distance_mat, n_samples, rss):
    indices = np.triu_indices(n_samples, k=1)
    distances = distance_mat[indices]
    tss = np.sum((distances - np.mean(distances))**2)

    return 1 - rss / tss

def compute_rss_clustering(distance_mat, clusters):
    clusters_unique = np.unique(clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in clusters_unique}

    intraclust_dists = []
    interclust_dists = {}
    rss = 0

    for c1 in clusters_unique:
        for c2 in clusters_unique:
            if c1 > c2: continue
            elif c1 == c2: # intra-cluster distances and RSS
                indices = cluster_to_indices[c1]
                intraclust_dist = distance_mat[np.ix_(indices, indices)]
                intraclust_dist = intraclust_dist[np.triu_indices(intraclust_dist.shape[0], k=1)]
                intraclust_dists.append(intraclust_dist)
                rss += np.sum((intraclust_dist - np.mean(intraclust_dist))**2)
            else: # inter-cluster distances and RSS
                indices1 = cluster_to_indices[c1]
                indices2 = cluster_to_indices[c2]
                interclust_dist = distance_mat[np.ix_(indices1, indices2)]
                interclust_dists[c1, c2] = interclust_dist
                rss += np.sum((interclust_dist - np.mean(interclust_dist))**2)
                
    return rss, intraclust_dists, interclust_dists
