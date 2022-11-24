import scipy.stats
import numpy as np
import pandas as pd

from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale


from sklearn.metrics import silhouette_score, silhouette_samples
from knac_helpers import split, prepareDf2

def split_into_clusters(X, Y, unique_labels):
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


# ////////////////////////////////////////////////

from sklearn.base import BaseEstimator, TransformerMixin

class KnacSplits(BaseEstimator, TransformerMixin):  
    # silhouette_metric={
    #                                'weight': 0.5, 
    #                                'data': X, 
    #                                'labels_automatic': Y, 
    #                                'labels_expert': E
    #                            }
    def __init__(self, 
                confidence_threshold=0.4, 
                silhouette_weight=None):
        # TODO: add some assert ()
        self.confidence_threshold = confidence_threshold
        self.silhouette_weight = silhouette_weight


    def fit(self, X, y=None, data=None, labels_automatic=None, labels_expert=None):
        self.data = data
        self.labels_automatic = labels_automatic
        self.labels_expert = labels_expert

        self.H = []
        for c in X.columns:
            entropy = scipy.stats.entropy(X[c])  # get entropy from counts
            self.H.append(entropy)
        
        self.H_split = normalize(X, axis=0)
        self.H_split = self.H_split * 1.0 / (np.array(self.H)/np.log2(len(X))+1) 
        self.H_split = pd.DataFrame(self.H_split, index=X.index, columns=X.columns)

        return self

    def transform(self, X):
        threshold = self.confidence_threshold
        use_silhouette = self.silhouette_weight is not None and self.silhouette_weight != 0
        
        if (use_silhouette):
            silhouette_metric_weight = self.silhouette_weight
            threshold = threshold * (1 - silhouette_metric_weight)

        to_split = self.H_split.apply(
            lambda x: [tuple((x[x > threshold]).index), np.mean(x[x > threshold])],
            axis=1)
        length = to_split.apply(lambda x: len(x[0]))
        split_recepie = to_split.to_frame(name='split')
        split_recepie['len'] = length
        splits_base = split_recepie[split_recepie['len']>1]['split']
        
        if (not use_silhouette):
            return splits_base
        
        data = self.data
        labels_automatic = self.labels_automatic
        labels_expert = self.labels_expert
        
        for row_idx, s in splits_base.items():
            silhouette_score_before = silhouette_score(data, labels_expert)
            
            cs = list(s[0])
            labels_expert_after_split = labels_expert
            for idx in range(1, len(cs)):
                labels_expert_after_split = split(labels_expert_after_split, labels_automatic, row=row_idx, col1=cs[0], col2=(cs[idx]))

            silhouette_score_after = silhouette_score(data, labels_expert_after_split)
            silhouette_diff = (silhouette_score_after - silhouette_score_before + 2) / 4
            base_confidence = s[1]
            s[1] = (1 - silhouette_metric_weight) * base_confidence + silhouette_metric_weight * silhouette_diff

        # splits_base = splits_base.where(splits_base[1] > self.confidence_threshold)
        # print(splits_base)
        # print(type(splits_base)) 
        
        return splits_base


# /////////////////////////////////////////////////


class KnacMerges(BaseEstimator, TransformerMixin):  
    # clusters_linkage_metric = {
    #         'weight': 0.2,
    #         'metric': metric, 
    #         'data': X, 
    #         'labels_expert': E            
    #     }
    def __init__(self, 
                confidence_threshold=0.8, 
                metric=None,
                metric_weight=0.2):
        # TODO: add some assert ()
        self.confidence_threshold = confidence_threshold
        self.metric = metric
        self.metric_weight = metric_weight
        
        self._clusters_linkage_metric_fns = {
            'single_link': single_link,
            'average_link': average_link,
            'complete_link': complete_link,
            'centroids_link': centroids_link,
            'wards_link': wards_link
        }

    def fit(self, X, y=None, data=None, labels_expert=None):
        self.data = data
        self.labels_expert = labels_expert

        self.H = []
        for c in X.columns:
            entropy = scipy.stats.entropy(X[c])  # get entropy from counts
            self.H.append(entropy)

        z = X.sum(axis=1)
        self.H_merge = X / z.values.reshape(-1, 1)
        self.H_merge = self.H_merge * (1 / (np.array(self.H) + 1))
        self.H_merge = self.H_merge.div(self.H_merge.sum(axis=1), axis=0)
        self.H_merge = pd.DataFrame(normalize(self.H_merge, axis=1), index=X.index, columns=X.columns)

        return self

    def transform(self, X):
        similarity_matrix = np.dot(self.H_merge.values, self.H_merge.values.T)
        
        if (self.metric is not None):
            metric = self.metric
            data = self.data
            labels_expert = self.labels_expert
            weight = self.metric_weight
            metric_fn = self._clusters_linkage_metric_fns[metric]
            
            clusters = split_into_clusters(data, labels_expert, X.index)
            distance_matrix = clusters_distance_matrix(clusters, metric_fn)
            distance_matrix = (distance_matrix - distance_matrix.min()) / distance_matrix.max()
            
            similarity_matrix = (1 - weight) * similarity_matrix + weight * (1 - distance_matrix)
        
        similarity = pd.DataFrame(similarity_matrix, index=self.H_merge.index, columns=self.H_merge.index)
        dists_fin = similarity[similarity > (self.confidence_threshold)].unstack().dropna().to_frame(name='similarity')
        dists_fin.index.set_names(["C1", "C2"], inplace=True)
        dists_fin = dists_fin.reset_index()
        dists_fin_nod = dists_fin[dists_fin['C1'] < dists_fin['C2']]

        return dists_fin_nod #.sort_values(by='similarity', ascending=False)
