import matplotlib.pyplot as plt
import seaborn as sns
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_moons

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np




def countData2(EC, E_cluster, C_cluster):
    return len(list(filter(lambda t: t[0] == E_cluster and t[1] == C_cluster, EC)))


def prepareDf2(E, C):
    EC = list(zip(E, C))
    dfData = {}
    
    E_clusters = np.sort(np.unique(E))
    C_clusters = np.sort(np.unique(C))
    
    for C_cluster in C_clusters:
        rowData = []
        for E_cluster in E_clusters:
            rowData.append(countData2(EC, E_cluster, C_cluster))
        
        dfData[str(C_cluster)] = rowData
    
    df = pd.DataFrame(dfData, index=E_clusters)
    return df


def find_centers(X, Y):
    XY = list(zip(X, Y))
    unique_labels = np.sort(np.unique(Y))
    clusters_no = len(unique_labels)
    features_no = len(X[0])
    
    clusters_data = dict()
    for label in unique_labels:
        clusters_data[label] = np.zeros(features_no)
        clusters_data[str(label) + '_count'] = 0
       
    for (x, label) in XY:
        clusters_data[label] = clusters_data[label] + x
        clusters_data[str(label) + '_count'] = clusters_data[str(label) + '_count'] + 1
        
    result = []
    for label in unique_labels:
        result.append(clusters_data[label] / clusters_data[str(label) + '_count'])
        
    return np.array(result)


def caption_clusters(ax, centers, scale=1):
    ax.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200*scale, edgecolor='k')
    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50*scale, edgecolor='k')

label_exper_clustering = 'Expert clustering'
label_automated_clustering = 'Automated clustering'
labels_heatmap = {
    'split': '$H_{split}$',
    'merge': '$H_{merge}$',
    'confusion': 'Confusion matrix'
}
label_heatmap_y_axis = 'E'
label_heatmap_x_axis = 'C'

        
def myPlot1(X, Y, E, dataset_name=None, centersY=None, centersE=None, patches=None, heatmap_matrix=None, heatmap_matrix_type=None, file=None):
    df = prepareDf2(E, Y) # deprecated
    
    le = LabelEncoder()
    E_integer_labels = le.fit_transform(E)
    Y_integer_labels = le.fit_transform(Y)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    ax0.set_title(label_exper_clustering)
    ax0.scatter(X[:, 0], X[:, 1], c=E_integer_labels, marker='o', s=100, cmap='viridis')
    if (centersE is not None):
        caption_clusters(ax0, centersE)
       
    if (patches is not None and 'E' in patches):
        for p in patches['E']:
            ax0.add_patch(p)
   
    ax1.set_title(label_automated_clustering)
    ax1.scatter(X[:, 0], X[:, 1], c=Y_integer_labels, marker='o', s=100, cmap='viridis')
    if (centersY is not None):
        caption_clusters(ax1, centersY)
        
    if (patches is not None and 'C' in patches):
        for p in patches['C']:
            ax1.add_patch(p)
        
    if (heatmap_matrix is None):
        heatmap_matrix = df
        
    sns.heatmap(np.around(heatmap_matrix, 3), annot=True, cmap='viridis',fmt='g',ax=ax2)
    ax2.set_ylim([heatmap_matrix.shape[0], 0]) # source: https://datascience.stackexchange.com/a/67741
    
    label_heatmap = labels_heatmap[heatmap_matrix_type]
    ax2.set_title(label_heatmap)
    plt.xlabel(label_heatmap_x_axis, fontsize = 11)
    plt.ylabel(label_heatmap_y_axis, fontsize = 11)
    
    if dataset_name:
        fig.suptitle('Dataset ' + dataset_name, y=1.03, x=0.08, fontsize=14)
        
    fig.tight_layout()

    if (file is not None):
        plt.savefig(file, dpi=150)

    plt.show()
    
    return df


def myPlot2(X, Y, E, dataset_name=None, biggerHeatmap=False, random_state=170, file=None):
    df = prepareDf2(E, Y)

    le = LabelEncoder()
    E_integer_labels = le.fit_transform(E)
    Y_integer_labels = le.fit_transform(Y)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

    X_ft = umap.UMAP(random_state=random_state).fit_transform(X)
    ax0.set_title(label_exper_clustering + ' (UMAP)')
    ax0.scatter(X_ft[:, 0], X_ft[:, 1], c=E_integer_labels)
    ax1.set_title(label_automated_clustering + ' (UMAP)')
    ax1.scatter(X_ft[:, 0], X_ft[:, 1], c=Y_integer_labels)
    
    if biggerHeatmap==False:
        sns.heatmap(np.around(df, 3), annot=True, cmap='viridis',fmt='g',ax=ax2)
        ax2.set_ylim([df.shape[0], 0]) # source: https://datascience.stackexchange.com/a/67741
        ax2.set_title(labels_heatmap['confusion'])
        plt.xlabel(label_heatmap_x_axis, fontsize = 11)
        plt.ylabel(label_heatmap_y_axis, fontsize = 11)
    else:
        ax2.axis('off')
        
    if dataset_name:
        fig.suptitle('Dataset ' + dataset_name, y=1.03, x=0.08, fontsize=14)
    
    fig.tight_layout()

    if (file is not None):
        plt.savefig(file, dpi=150)

    plt.show()
    
    if biggerHeatmap==True:
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(np.around(df, 3), annot=True, cmap='viridis',fmt='g', ax=ax)
        ax.set_ylim([df.shape[0], 0]) # source: https://datascience.stackexchange.com/a/67741
        ax.set_title(labels_heatmap['confusion'])
        plt.xlabel(label_heatmap_x_axis, fontsize = 11)
        plt.ylabel(label_heatmap_y_axis, fontsize = 11)
    
    return df


def myPlot3D(X, Y, E, dataset_name=None):
    df = prepareDf2(E, Y)

    fig = make_subplots(rows=1, cols=3, 
                        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'xy'}]],
                        subplot_titles=(label_exper_clustering, label_automated_clustering, labels_heatmap['confusion']))

    fig.add_trace(
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], 
                                       showlegend=False, 
                                       mode='markers', 
                                       marker=dict( 
                                         color=E, 
                                         colorscale ='Viridis',
                                       )),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], 
                                       showlegend=False,
                                       mode='markers', 
                                       marker=dict( 
                                         color=Y, 
                                         colorscale ='Viridis',
                                       )),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(
            x = list(df.columns),
            y = list(df.index),
            z=df.to_numpy().astype(int),
            type = 'heatmap',
            colorscale = 'Viridis'
        ),
        row=1, col=3
    )
    
    title_text = 'Dataset ' + dataset_name if dataset_name else ''

    fig.update_layout(height=400, width=900, title_text=title_text)
    fig.show()
    
    return df


# -----------------------------------------------


def make_blobs_weights(weights, **args):
    generate_centers = 'return_centers' in args and args['return_centers'] is True
    centers = None
    if (generate_centers):
        X, Y, centers = make_blobs(**{**args, 'n_samples': args['n_samples'] * args['centers']})
    else:
        X, Y = make_blobs(**{**args, 'n_samples': args['n_samples'] * args['centers']})
    
    clusters = []
    clusters_labels = []
    points = []
    
    for idx in range(args['centers']):
        clusters.append([])
    
    for idx in range(len(Y)):
        clusters[Y[idx]].append(X[idx])
        
    for cluster_no in range(len(clusters)):
        cluster_data = np.array(clusters[cluster_no])
        size = int(args['n_samples'] * weights[cluster_no])
        clusters[cluster_no] = cluster_data[np.random.choice(len(cluster_data), size=size, replace=False)]
        clusters_labels.extend([cluster_no] * len(clusters[cluster_no]))
        points.extend(clusters[cluster_no])

    if (generate_centers):
        return (np.array(points), clusters_labels, centers)
    else:
        return (np.array(points), clusters_labels)

def generate_weights(size):
    weights = np.random.randint(1, 10, size=size)
    weights = weights / weights.sum()
    
    return list(weights)


# -----------------------------------------------


from IPython.display import display, display_html

def display_side_by_side(*args): # source: https://stackoverflow.com/a/44923103/5766602
    html_str=''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'), raw = True)


# -----------------------------------------------


from tabulate import tabulate

expert_cluster1_name = 'exp. cluster 1'
expert_cluster2_name = 'exp. cluster 2'

def print_merges(merges):
    print(tabulate(
        merges.rename(columns = {'Cluster1': expert_cluster1_name, 'Cluster2': expert_cluster2_name}, inplace = False), 
        tablefmt='pipe', headers='keys', showindex=False))

def print_splits(splits):
    transform_clusters_to_ints = lambda clusters: map(lambda c: int(c[1:]) - 1, clusters)
    transform_clusters = lambda clusters: ', '.join(map(str, tuple(transform_clusters_to_ints(clusters))))
    splits2 = splits.apply(lambda row: list([transform_clusters(row[0]), row[1]]))
    print(tabulate(splits2, headers = ['exp. cluster', 'alg. clusters', 'confidence'], tablefmt='pipe'))

    
    
# -----------------------------------------------


    
def merge(E, row_idx1, row_idx2):
    return np.where(E == row_idx2, row_idx1, E)
    
def split(E, Y, row, col1, col2):
    mask = np.invert(np.logical_and(E == row, Y == col2))
    #new_cluster_no = np.amax(E) + 1
    new_cluster_no = f'split->{row};{col1};{col2}'
    return np.where(mask, E, new_cluster_no)  
