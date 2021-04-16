
import numpy as np
import scipy.sparse as sp
import os.path as osp



def load_train_test_data(dataset):
    try:
        base_dir = osp.dirname(osp.realpath(__file__))
        ts_train = np.loadtxt(osp.join(base_dir, "data", dataset + "_TRAIN.txt"))
        ts_test = np.loadtxt(osp.join(base_dir, "data", dataset + "_TEST.txt"))
    except IOError:
        raise ValueError("Could not find dataset: " + dataset) from None
    return ts_train, ts_test

def load_coalesced_data(dataset):
    ts_train, ts_test = load_train_test_data(dataset)
    
    labels = np.hstack((ts_train[:,0], ts_test[:,0])).astype(int)
    _, labels = np.unique(labels, return_inverse=True)
    features = np.vstack((ts_train[:,1:], ts_test[:,1:]))
    
    return features, labels, ts_train.shape[0]


def shift_and_scale(features):
    features -= features.mean(axis=1, keepdims=True)
    features /= features.std(axis=1, ddof=1, keepdims=True)
    return features

def distance_matrix_dtw(features):
    
    from fastdtw import fastdtw

    features = shift_and_scale(features)
    num_samples = features.shape[0]
    
    distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i):
            distances[i,j] = distances[j,i] = fastdtw(features[i], features[j])[0]
    
    return distances

def distance_matrix_sdtw(features, gamma=0.1, correction=True):
    
    features = shift_and_scale(features)
    num_samples = features.shape[0]
    
    from sdtw import SoftDTW
    from sdtw.distance import SquaredEuclidean
    
    def dist(X, Y):
        return SoftDTW(SquaredEuclidean(X.reshape(-1,1), Y.reshape(-1,1)), gamma).compute()
    
    if correction:
        self_distances = [dist(x,x) for x in features]
    
    distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i):
            value = dist(features[i], features[j])
            if correction:
                value -= 0.5*(self_distances[i] + self_distances[j])
            distances[i,j] = distances[j,i] = value

    return distances


def distance_matrix_mpdist(features, window_length=None, percentage=0.05):
    
    import stumpy
    
    features = shift_and_scale(features)
    num_samples = features.shape[0]
    
    if window_length is None:
        window_length = features.shape[1] // 2
    
    distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i):
            value = stumpy.mpdist(features[i], features[j], m=window_length, percentage=percentage, k=None)
            distances[i,j] = distances[j,i] = value
    
    return distances


def distance_matrix_euclidean(features):
    
    import scipy.spatial.distance as spdist
    
    features = shift_and_scale(features)
    
    return spdist.squareform(spdist.pdist(features, metric='euclidean'))
    

def fixed_adjacency(distances, sigma):
    return np.exp(- distances**2 / sigma**2)

def self_tuning_adjacency(distances, knn=7):
    inv_sigmas = 1 / np.partition(distances, knn, axis=0)[knn]
    return np.exp(- inv_sigmas[:,np.newaxis] * (distances**2) * inv_sigmas)


# def graph_laplacian_eigs(adjacency, num_ev):
#     degrees = adjacency.sum(0)
#     adjacency = np.diagflat(degrees) - adjacency

def normalized_laplacian_eigs(adjacency, num_ev, economy=False):
    degrees = 1 / np.sqrt(adjacency.sum(0))
    
    adjacency = degrees[:,np.newaxis] * adjacency * degrees
    
    if num_ev is None:
        w, U = np.linalg.eigh(adjacency)
        w = 1 - w
    else:
        w, U = sp.linalg.eigsh(adjacency + np.eye(adjacency.shape[0]), k=num_ev)
        w = 2 - w
    
    ind = np.argsort(w)
    w = w[ind]
    U = U[:,ind]
    return w, U
