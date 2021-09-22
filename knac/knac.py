import scipy.stats
import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale


from sklearn.metrics import silhouette_score, silhouette_samples
from knac_helpers import split

def split_into_clusters(X, Y):
    unique_labels = np.unique(Y)
    clusters_n = len(unique_labels)
    clusters = [0] * clusters_n
    
    labels_cluster_indexes = {}
    for no, label in enumerate(unique_labels):
        labels_cluster_indexes[label] = no
    
    for i in range(clusters_n):
        clusters[i] = []
        
    for i ,(x, label) in enumerate(zip(X, Y)):
        clusters[labels_cluster_indexes[label]].append(x)
        
    return clusters#[c for c in clusters if len(c)>0]

def clusters_distance_matrix(clusters, metric):
    clusters_n = len(clusters)
    dists = np.empty([clusters_n, clusters_n])
    for i in range(1, clusters_n):
        for j in range(0, i):
            dists[i][j] = metric(clusters[i], clusters[j])
            dists[j][i] = dists[i][j]
            
        # diagonal:
        dists[i][i] = metric(clusters[i], clusters[i])
        
    dists[0][0] = metric(clusters[0], clusters[0])
        
    return dists

def single_link(c1, c2):
    dists = distance.cdist(c1, c2, 'euclidean')
    return dists.min()

def complete_link(c1, c2):
    dists = distance.cdist(c1, c2, 'euclidean')
    return dists.max()

def average_link(c1, c2):
    dists = distance.cdist(c1, c2, 'euclidean')
    return dists.mean()

def centroids_link(c1, c2):
    length1 = len(c1)# .shape[0]
    length2 = len(c2)#.shape[0]
    centroid1 = np.sum(c1, axis=0) / length1
    centroid2 = np.sum(c2, axis=0) / length2
    
    return distance.euclidean(centroid1, centroid2)

def wards_link(c1, c2):
    combined_clusters = np.vstack((c1, c2))
    length = combined_clusters.shape[0]
    centroid = np.sum(combined_clusters, axis=0) / length
    
    def centroid_std(x):
        dist = np.linalg.norm(x - centroid)
        return dist

    dists = np.apply_along_axis(centroid_std, 1, combined_clusters)
    return dists.sum()

class KNAC:
    def __init__(
            self, 
            split_confidence_threshold=0.3, 
            merge_confidence_threshold=0.8):
        self.split_confidence_threshold = split_confidence_threshold
        self.merge_confidence_threshold = merge_confidence_threshold
        self.H = None
        self.H_conf = None
        self.H_conf_n = None
        
        self.clusters_linkage_metric_fns = {
            'single_link': single_link,
            'average_link': average_link,
            'complete_link': complete_link,
            'centroids_link': centroids_link,
            'wards_link': wards_link
        }

        
        
    def fit(self,X):
        self.H = []
        for c in X.columns:
            entropy = scipy.stats.entropy(X[c])  # get entropy from counts
            self.H.append(entropy)
        
        self.H_conf2 = normalize(X, axis=0)
        self.H_conf2 = self.H_conf2 * 1.0 / (np.array(self.H)/np.log2(len(X))+1) 
        #self.H_conf2 = minmax_scale(self.H_conf2, axis=1)
        self.H_conf2 = pd.DataFrame(self.H_conf2, index=X.index, columns=X.columns)

        z = X.sum(axis=1)
        self.H_conf = X / z.values.reshape(-1, 1)
        self.H_conf = self.H_conf * (1 / (np.array(self.H) + 1))
        self.H_conf = self.H_conf.div(self.H_conf.sum(axis=1), axis=0)
        self.H_conf_n = pd.DataFrame(normalize(X, axis=1), index=X.index, columns=X.columns)

        return self

#     silhouette_metric={'weight': 0.5, 'data': X, 'labels_automatic': Y, 'labels_expert': E })
    def splits(self, threshold_override=None, silhouette_metric=None):
        threshold = self.split_confidence_threshold if threshold_override is None else threshold_override
        
        if (silhouette_metric is not None):
            silhouette_metric_weight = silhouette_metric['weight']
            threshold = threshold * (1 - silhouette_metric_weight)

        to_split = self.H_conf2.apply(
            lambda x: [tuple((x[x > threshold]).index), np.mean(x[x > threshold])],
            axis=1)
        length = to_split.apply(lambda x: len(x[0]))
        split_recepie = to_split.to_frame(name='split')
        split_recepie['len'] = length
        splits_base = split_recepie[split_recepie['len']>1]['split']
        
        if (silhouette_metric is None):
            return splits_base
        
        data = silhouette_metric['data']
        labels_automatic = silhouette_metric['labels_automatic']
        labels_expert = silhouette_metric['labels_expert']
        
        for row_idx, s in splits_base.items():
            silhouette_score_before = silhouette_score(data, labels_expert)
            
            cs = list(map(lambda s: int(s), s[0]))
            labels_expert_after_split = labels_expert
            for idx in range(1, len(cs)):
                labels_expert_after_split = split(labels_expert_after_split, labels_automatic, row=row_idx, col1=cs[0], col2=(cs[idx]))

            silhouette_score_after = silhouette_score(data, labels_expert_after_split)
            silhouette_diff = (silhouette_score_after - silhouette_score_before + 2) / 4
            base_confidence = s[1]
            s[1] = (1 - silhouette_metric_weight) * base_confidence + silhouette_metric_weight * silhouette_diff
        
        return splits_base

#     clusters_linkage_metric={'metric': metric, 'data': X, 'labels_expert': E, 'weight': 0.2}
    def merges(self, clusters_linkage_metric=None):
        similarity_matrix = np.dot(self.H_conf_n.values, self.H_conf_n.values.T)
        
        if (clusters_linkage_metric is not None):
            metric = clusters_linkage_metric['metric']
            data = clusters_linkage_metric['data']
            labels_expert = clusters_linkage_metric['labels_expert']
            weight = clusters_linkage_metric['weight']
            metric_fn = self.clusters_linkage_metric_fns[metric]
            
            clusters = split_into_clusters(data, labels_expert)
            distance_matrix = clusters_distance_matrix(clusters, metric_fn)
            distance_matrix = (distance_matrix - distance_matrix.min()) / distance_matrix.max()
            
            similarity_matrix = (1 - weight) * similarity_matrix + weight * (1 - distance_matrix)
        
        similarity = pd.DataFrame(similarity_matrix, index=self.H_conf_n.index, columns=self.H_conf_n.index)
        dists_fin = similarity[similarity > (self.merge_confidence_threshold)].unstack().dropna().to_frame(name='similarity')
        dists_fin.index.set_names(["C1", "C2"], inplace=True)
        dists_fin = dists_fin.reset_index()
        dists_fin_nod = dists_fin[dists_fin['C1'] < dists_fin['C2']]

        return dists_fin_nod
