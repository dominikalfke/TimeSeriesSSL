

import numpy as np
import utils
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphConvolutionalNet(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden=32, dropout=0.5, bias=True, normalize=True, loops=False):
        
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, cached=True, bias=bias, normalize=normalize, add_self_loops=loops)
        self.conv2 = GCNConv(hidden, out_channels, cached=True, bias=bias, normalize=normalize, add_self_loops=loops)
        self.dropout = dropout

    def forward(self, data):
        edge_index, edge_weight = data['edge_index'], data['edge_weight']
            
        x = data.x
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    
    def split_parameters(self, weight_decay=5e-4):
        return [
            dict(params=self.conv1.parameters(), weight_decay=weight_decay),
            dict(params=self.conv2.parameters(), weight_decay=0)
        ]
    
    def run_training(self, data, optimizer, num_epochs=200):
        self.train()
        
        y = data.y[data.train_mask]
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = self(data)
            loss = F.cross_entropy(out[data.train_mask, :], y)
            loss.backward()
            optimizer.step()
        return loss.item()


    def predict(self, data, mask=slice(None)):
        return self(data)[mask].max(dim=1)[1]
        
    def eval_accuracy(self, data):
        self.eval()
        correctness = self.predict(data, data.test_mask).eq(data.y[data.test_mask])
        return float(correctness.sum().item()) / data.test_mask.sum().item()


if __name__ == '__main__':
    
    
    import argparse
    from time import perf_counter as timer
    from datetime import datetime

    import torch
    from torch_geometric.data import Data
    import aux
    
    parser = argparse.ArgumentParser(description='...')
    
    parser.add_argument('dataset', help='Name of the dataset')
    
    parser.add_argument('--hidden', type=int, default=32, help='Hidden layer width')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--loops', action='store_true', default=False, help='Add self loops')
    
    parser.add_argument('-n', '--num-runs', type=int, default=1, metavar='RUNS', help='Number of experimental runs')
    parser.add_argument('-s', '--random-splits', type=int, default=None, metavar='SPLIT_SIZE', help='Use different random training/test splits with S training samples per class')
    parser.add_argument('--relative-random-splits', type=float, default=None, metavar='SPLIT_RATIO', help='Like --random-splits, but specifying a relative ratio of training samples')
    
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for parameter training')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay for parameter training')
    
    parser.add_argument('-S', '--sparsify', type=int, default=None, metavar='K', help='Only use a sparsified adjacency matrix of the top K neighbors')
    parser.add_argument('-R', '--reuse', action='store_true', default=False, help='Load existing distance matrix or store it after computation')
    
    parser.add_argument('--distance', choices=['DTW', 'SDTW', 'MPDist', 'Euclidean'], default='DTW', help='Strategy for distance computation')
    parser.add_argument('--sdtw-gamma', type=float, default=1.0, help='Gamma parameters for distance computation strategy SDTW')
    parser.add_argument('--fixed-sigma', type=float, default=None, help='Use a fixed sigma instead of self tuning')
    parser.add_argument('--knn', type=int, default=7, help='Index of nearest neighbor used for self-tuning sigma')
    
    parser.add_argument('--disable-cuda', action='store_true', default=False, help='Turn off GPU acceleration')
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
    
    if args.sparsify is not None and args.sparsify >= num_samples-1:
        print("Chosen sparsification {} is larger or equal to the maximum neighbor count {}. Proceeding with maximum value.".format(
            args.sparsify, num_samples-1))
        args.sparsify = None
    
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
        
    ## Compute adjacency matrix
    tic = timer()
    if args.fixed_sigma is not None:
        adjacency = utils.fixed_adjacency(distances, args.fixed_sigma)
    else:
        adjacency = utils.self_tuning_adjacency(distances, args.knn)
    time_adjacency = timer() - tic
    log("Time for adjacency setup using {}: {:.2f} seconds",
        "sigma={}".format(args.fixed_sigma) if args.fixed_sigma is not None else "knn={}".format(args.knn), 
        time_adjacency)
    del distances
    
    ### Transform into sparse format
    tic = timer()
    edge_index = []
    edge_weight = []
    
    S = args.sparsify
    for i in range(0,num_samples):
        if S is None:
            if i > 0:
                edge_index.append(np.vstack(([i]*i, np.arange(i))))
                edge_weight.append(adjacency[i,:i])
            if i < num_samples-1:
                edge_index.append(np.vstack(([i]*(num_samples-i-1), np.arange(i+1,num_samples))))
                edge_weight.append(adjacency[i,i+1:])
        else:
            ind = np.argpartition(adjacency[i], -S-1)[-S-1:]
            ind_not_i = ind != i
            ind = ind[1:] if all(ind_not_i) else ind[ind_not_i]
            
            edge_index.append(np.vstack(([i]*S, ind)))
            edge_weight.append(adjacency[i, ind])
            
            
    edge_index = np.hstack(edge_index)
    edge_weight = np.hstack(edge_weight)
    time_transform = timer() - tic
    log("Time for format transformation: {:.2f}", time_transform)
    del adjacency
    
    ### Setup data for torch
    
    tic = timer()
        
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    train_mask[:training_size] = True
    
    data = Data(x = torch.FloatTensor(features),
                y = torch.LongTensor(labels),
                train_mask = train_mask,
                test_mask = ~train_mask,
                edge_index = torch.LongTensor(edge_index),
                edge_weight = torch.FloatTensor(edge_weight))
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
    data = data.to(device)

    model = GraphConvolutionalNet(num_features, num_classes,
                                  hidden=args.hidden, dropout=args.dropout, loops=args.loops)
                                  
    model.to(device)
    
    optimizer = torch.optim.Adam(model.split_parameters(args.weight_decay), lr=args.learning_rate)
    
    
    if args.relative_random_splits is not None:
        split_size = int(np.ceil(args.relative_random_splits * num_samples / num_classes))
    else:
        split_size = args.random_splits
    
    
    time_model = timer() - tic
    log('Time for model setup: {:.2f} seconds', time_model)
        
    time_setup = time_adjacency + time_transform + time_model
    log('Setup done in {:.2f} seconds', time_setup)
    
    time_training = []
    accuracies = []
    
    for run in range(args.num_runs):
        aux.set_seed(run)
    
        if split_size is not None:
            data.train_mask = torch.zeros(data.num_nodes, dtype=bool, device=device)
            for c in range(num_classes):
                ind = torch.nonzero(data.y == c).flatten()
                ind = ind[torch.randperm(len(ind), device=device)[:split_size]]
                data.train_mask[ind] = True
            data.test_mask = ~data.train_mask
    
        tic = timer()
        model.reset_parameters()
        
        model.run_training(data, optimizer, num_epochs=args.epochs)
        
        t = timer() - tic
        time_training.append(t)
    
        acc = model.eval_accuracy(data)
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
        
        architecture_name = 'StandardGCN'
        if args.loops:
            architecture_name += '_loops'
        if args.sparsify is not None:
            architecture_name += '_sparsify{}'.format(args.sparsify)
        architecture_name += '_' + strategy_name
        if args.knn != 7:
            architecture_name += '_knn{}'.format(args.knn)
        
        aux.save_results(
            results_dir, dataset_name, architecture_name,
            accuracies, time_setup, time_training,
            {
             'Distance matrix': 'Loaded from {}'.format(dist_matrix_filename) if time_distances == 0 else 'Computed in {:.2f} seconds'.format(time_distances),
             'Start time': start_time.strftime("%b %d, %Y, %H:%M:%S"),
             'End time': datetime.now().strftime("%b %d, %Y, %H:%M:%S")
            },
            args.__dict__, 
            split_size = split_size,
            file = __file__)
