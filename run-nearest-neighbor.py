
import numpy as np
import utils


def one_nearest_neighbor(distances, labels):
    labels = labels.copy()
    ind_labeled = np.flatnonzero(labels >= -1)
    ind_unlabeled = np.flatnonzero(labels < 0)
    
    np.fill_diagonal(distances, np.inf)
    
    while len(ind_unlabeled) > 0:
        min_dist = np.inf
        for irow in ind_unlabeled:
            for icol in ind_labeled:
                if distances[irow,icol] < min_dist:
                    (ind1,ind2) = (irow,icol)
                    min_dist = distances[irow,icol]
                    
        labels[ind1] = labels[ind2]
        
        ind_unlabeled = ind_unlabeled[ind_unlabeled != ind1]
        ind_labeled = np.append(ind_labeled, ind1)
        
    return labels

def one_nearest_neighbor_new(distances, labels):
    labels = labels.copy()
    ind_labeled = np.flatnonzero(labels >= 0)
    # ind_unlabeled = np.flatnonzero(labels < 0)
    
    np.fill_diagonal(distances, np.inf)
    
    # nearest_labeled_neighbors = ind_labeled[np.argmin(distances[ind_unlabeled][:, ind_labeled], axis=1)]
    # nearest_labeled_distances = [distances[ind_unlabeled[i], j] for i, j in enumerate(nearest_labeled_neighbors)]
    
    # while len(ind_unlabeled) > 0:
    #     entry = np.argmin(nearest_labeled_distances)
    #     new_labeled_index = ind_unlabeled[entry]
    #     labels[new_labeled_index] = labels[nearest_labeled_neighbors[entry]]
        
    #     if len(ind_unlabeled) == 1:
    #         break
        
    #     nearest_labeled_neighbors = np.delete(nearest_labeled_neighbors, entry)
    #     nearest_labeled_distances = np.delete(nearest_labeled_distances, entry)
    #     ind_unlabeled = np.delete(ind_unlabeled, entry)
    #     ind_labeled = np.append(ind_labeled, new_labeled_index)
        
    #     for entry, unlabeled_index in enumerate(ind_unlabeled):
    #         d = distances[unlabeled_index, new_labeled_index]
    #         if d < nearest_labeled_distances[entry]:
    #             nearest_labeled_distances[entry] = d
    #             nearest_labeled_neighbors[entry] = new_labeled_index
    
    
    # print("Number of labeled nodes: {}/{}".format(len(ind_labeled), labels.size))
    
    import heapq
    heap = []
    for i in range(labels.size):
        if labels[i] < 0:
            j = ind_labeled[np.argmin(distances[i,ind_labeled])]
            heap.append((distances[i,j], i, j))
    heapq.heapify(heap)
    
    while len(heap) > 0:
        _, i, j = heapq.heappop(heap)
        # print("Closest pair: Unlabeled {}, labeled {}".format(i, j))
        labels[i] = labels[j]
        
        for entry in range(len(heap)):
            old_dist, j, _ = heap[entry]
            new_dist = distances[i,j]
            if new_dist < old_dist:
                heap[entry] = (new_dist, j, i)
        
        heapq.heapify(heap)
    
    
    return labels


if __name__ == '__main__':
    
    
    import argparse
    from time import perf_counter as timer
    from datetime import datetime

    import aux
    
    parser = argparse.ArgumentParser(description='...')
    
    parser.add_argument('dataset', help='Name of the dataset')
    
    
    parser.add_argument('-n', '--num-runs', type=int, default=1, metavar='RUNS', help='Number of experimental runs')
    parser.add_argument('-s', '--random-splits', type=int, default=None, metavar='SPLIT_SIZE', help='Use different random training/test splits with S training samples per class')
    parser.add_argument('--relative-random-splits', type=float, default=None, metavar='SPLIT_RATIO', help='Like --random-splits, but specifying a relative ratio of training samples')
    
    # parser.add_argument('-S', '--sparsify', type=int, default=None, metavar='K', help='Only use a sparsified adjacency matrix of the top K neighbors')
    parser.add_argument('-R', '--reuse', action='store_true', default=False, help='Load existing distance matrix or store it after computation')
    
    parser.add_argument('--distance', choices=['DTW', 'SDTW', 'MPDist', 'Euclidean'], default='Euclidean', help='Strategy for distance computation')
    parser.add_argument('--sdtw-gamma', type=float, default=1.0, help='Gamma parameters for distance computation strategy SDTW')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print information during execution')
    parser.add_argument('--no-save', action='store_true', default=False, help='Disable saving the results')
    
    args = parser.parse_args()

    start_time = datetime.now()
    
    def log(m, *a, **kw):
        if args.verbose:
            print(m.format(*a, **kw))
    
    ## Load data
    
    features, labels, training_size = utils.load_coalesced_data(args.dataset)
    
    num_samples, num_features = features.shape
    test_size = num_samples - training_size
    num_classes = labels.max()+1
    
    log("Loaded {} samples ({} + {}) with {} features from {} classes", num_samples, training_size, test_size, num_features, num_classes)
    
    
    ## Compute distance matrix

    strategy_name = args.distance
    if strategy_name == 'SDTW':
        strategy_name += "_gamma{:g}".format(args.sdtw_gamma)
    
    distances = None
    time_distances = 0
    if args.reuse:
        dist_matrix_filename = "./matrices/{}_{}.npy".format(args.dataset, strategy_name)
        try:
            distances = np.load(dist_matrix_filename)
            log("Distance matrix loaded from {}", dist_matrix_filename)
        except IOError:
            log("Tried to reuse distance matrix, but failed to load from {}", dist_matrix_filename)
    
    if distances is None:
        tic = timer()
        if args.distance == 'DTW':
            distances = utils.distance_matrix_dtw(features)
        elif args.distance == 'SDTW':
            distances = utils.distance_matrix_sdtw(features, args.sdtw_gamma)
        elif args.distance == 'MPDist':
            distances = utils.distance_matrix_mpdist(features)
        elif args.distance == 'Euclidean':
            distances = utils.distance_matrix_euclidean(features)
        else:
            raise ValueError('Unknown distance computation strategy: {}'.format(args.distance))
        time_distances = timer() - tic
        log("Time for distance matrix setup using {} strategy: {:.2f} seconds", args.distance, time_distances)
        if args.reuse:
            np.save(dist_matrix_filename, distances)
            log("Distance matrix saved to {}", dist_matrix_filename)
        
    
    ### Setup data
    
    tic = timer()
    
    if args.relative_random_splits is not None:
        split_size = int(np.ceil(args.relative_random_splits * num_samples / num_classes))
    else:
        split_size = args.random_splits
    
    if split_size is None:
        train_mask = np.zeros(num_samples, dtype=bool)
        train_mask[:training_size] = True
        
        y = -np.ones(num_samples)
        y[train_mask] = labels[train_mask]
    else:
        training_size = num_classes * split_size
    
    test_size = num_samples - training_size
    
    time_model = timer() - tic
    log('Time for model setup: {:.2f} seconds', time_model)
        
    time_setup = time_distances + time_model
    log('Setup done in {:.2f} seconds', time_setup)
    
    time_training = []
    accuracies = []
    
    for run in range(args.num_runs):
        aux.set_seed(run)
    
        if split_size is not None:
            train_mask = np.zeros(num_samples, dtype=bool)
            for i in range(num_classes):
                label_mask = labels == i
                ind = np.random.choice(range(num_samples), split_size, replace=False,
                                       p=label_mask/label_mask.sum())
                train_mask[ind] = True
            
            y = -np.ones(num_samples)
            y[train_mask] = labels[train_mask]
    
        tic = timer()
        
        out = one_nearest_neighbor_new(distances, y)
        
        t = timer() - tic
        time_training.append(t)
        
        
        acc = np.mean(out[~train_mask] == labels[~train_mask])
        accuracies.append(acc)
        log('Run {}/{}: Training time {:.4f} s, accuracy {:.4%}', run+1, args.num_runs, t, acc)
    
    print('###')
    aux.print_results(accuracies, time_setup, time_training)
    print('###')

    if not args.no_save:
        results_dir = './results'
        
        dataset_name = args.dataset
        if split_size is not None:
            dataset_name += '_split{}'.format(split_size)
        
        architecture_name = 'NearestNeighbor'
        architecture_name += '_' + strategy_name
        
        aux.save_results(
            results_dir, dataset_name, architecture_name,
            accuracies, time_setup, time_training,
            {
             'Distance matrix': 'Loaded from {}'.format(dist_matrix_filename) if time_distances == 0 else 'Computed in {:.2f} seconds'.format(time_distances),
             'Start time': start_time.strftime("%b %d, %Y, %H:%M:%S"),
             'End time': datetime.now().strftime("%b %d, %Y, %H:%M:%S")
            },
            args.__dict__, 
            file = __file__)
