
import numpy as np
import utils

# def convexity_splitting(u0, Lambda, phi,omega_0,epsilon,dt,c,MAX_ITER,tolit):
    
#     # set size parameters
#     n = u0.size
    
#     omega = np.zeros(n) #penalty parameter for fidelty term
#     omega[u0!=0]=omega_0 #penalty parameter for fidelty term
    
    
    
#     ## Convexity splitting 
#     a = phi.T @ u0
#     u = phi @ a
#     D = 1 / (1/dt + epsilon*Lambda + c)
    
#     for x in range(0,MAX_ITER):
#        a  = D*((1/dt+c)*a + phi.T @ (-4*(u**3 - u)/epsilon + omega*(u0-u)))
#        u_new = phi @ a
#        if np.linalg.norm(u_new-u) < tolit*np.linalg.norm(u):
#           break
    
#        u = u_new
    
#     return u

class ConvexitySplitting:
    
    def __init__(self, penalty=1e10, epsilon=None, c=None, max_iter=500, time_step=0.01, tolerance=1e-8):
        self.penalty = penalty
        self.epsilon = epsilon
        self.c = c
        self.max_iter = max_iter
        self.time_step = time_step
        self.tolerance = tolerance
    
    def run(self, u0, ev, phi):
        n, k = phi.shape
        
        omega0 = self.penalty
        epsilon = self.epsilon if self.epsilon is not None else 1.0 / np.sqrt(n)
        c_plus_inv_dt = (self.c if self.c is not None else 3/epsilon + omega0) + 1 / self.time_step
        
        # diagonal of penalty matrix Omega
        omega = np.zeros(n)
        omega[u0 != 0] = omega0
        
        # diagonal of matrix multiplied from the left
        D = 1 / (epsilon*ev + c_plus_inv_dt)
        
        # a: low-rank coefficients, u: full-rank solution - u = phi @ a
        a = phi.T @ u0
        # u = phi @ a
        u = u0 # not exactly correct, but close
        
        for it in range(self.max_iter):
            a = D * (c_plus_inv_dt*a + phi.T @ (-4*epsilon*(u**3 - u)/epsilon + omega*(u0 - u)))
            u_old = u
            u = phi @ a
            
            if np.linalg.norm(u - u_old) < self.tolerance * np.linalg.norm(u_old):
                break
        
        return u
    
    # def run_legacy(self, u0, ev, phi):
    #     n,k = phi.shape
    #     epsilon = self.epsilon if self.epsilon is not None else 1.0 / np.sqrt(n)
    #     c = self.c if self.c is not None else 3/epsilon + self.penalty
    #     u = convexity_splitting(u0, ev, phi, self.penalty, epsilon, self.time_step, c, self.max_iter, self.tolerance)
    #     print("u.shape: ", u.shape)
    #     return u
        

if __name__ == '__main__':
    
    
    import argparse
    from time import perf_counter as timer
    from datetime import datetime

    import aux
    
    parser = argparse.ArgumentParser(description='...')
    
    parser.add_argument('dataset', help='Name of the dataset')
    
    parser.add_argument('-r', '--rank', type=int, default=None, help='Number of eigenvalues used (default: n)')
    parser.add_argument('--skip-ev', type=int, default=0, nargs='?', const=1, metavar='K', help='Skip the first one or K eigenvalues')
    parser.add_argument('--economy', action='store_true', default=False, help='Do not compute the full eigenvalue decomposition')
    
    parser.add_argument('--penalty', type=float, default=1e10, help='Penalty coefficient omega0 (default: 1e10)')
    parser.add_argument('--epsilon', type=float, default=None, help='Parameter epsilon for interface width or something (default: 1/sqrt(n))')
    parser.add_argument('--c-splitting', type=float, default=None, help='Paramteter c for convexity splitting (default: 3/epsilon+omega0)')
    parser.add_argument('--time-step', type=float, default=0.01, help='Time step for ODE solver (default: 0.01)')
    parser.add_argument('--max-iter', type=int, default=500, help='Maximum number of iterations (default: 500)')
    parser.add_argument('--tolerance', type=float, default=1e-8, help='Tolerance for early stopping (default: 1e-8)')
    
    
    parser.add_argument('-n', '--num-runs', type=int, default=1, metavar='RUNS', help='Number of experimental runs')
    parser.add_argument('-s', '--random-splits', type=int, default=None, metavar='SPLIT_SIZE', help='Use different random training/test splits with S training samples per class')
    parser.add_argument('--relative-random-splits', type=float, default=None, metavar='SPLIT_RATIO', help='Like --random-splits, but specifying a relative ratio of training samples')
    
    # parser.add_argument('-S', '--sparsify', type=int, default=None, metavar='K', help='Only use a sparsified adjacency matrix of the top K neighbors')
    parser.add_argument('-R', '--reuse', action='store_true', default=False, help='Load existing distance matrix or store it after computation')
    
    parser.add_argument('--distance', choices=['DTW', 'SDTW', 'MPDist', 'Euclidean'], default='DTW', help='Strategy for distance computation')
    parser.add_argument('--sdtw-gamma', type=float, default=1.0, help='Gamma parameters for distance computation strategy SDTW')
    parser.add_argument('--fixed-sigma', type=float, default=None, help='Use a fixed sigma instead of self tuning')
    parser.add_argument('--knn', type=int, default=7, help='Index of nearest neighbor used for self-tuning sigma')
    
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
    
    if num_classes != 2:
        raise ValueError("Allen Cahn can currently only be used for binary classification")
    labels[labels == 0] = -1
    
    if args.rank is not None and args.rank+args.skip_ev >= num_samples:
        print("Chosen rank {} is larger or equal to the maximum rank {}. Proceeding with maximum rank.".format(
            args.rank, num_samples-args.skip_ev))
        args.rank = None
    
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
    
    
    ## Compute graph Laplacian eigenvalues
    tic = timer()
    num_ev = args.rank+args.skip_ev if args.rank is not None and args.rank+args.skip_ev < num_samples else None
    w, U = utils.normalized_laplacian_eigs(adjacency, num_ev if args.economy else None)
    
    ind = slice(args.skip_ev, num_ev)
    w = w[ind]
    U = U[:, ind]
    
    time_eigs = timer() - tic
    log("Time for computation of {} eigenvalues: {:.2f} seconds", "all" if num_ev is None or not args.economy else num_ev, time_eigs)
    del adjacency
    
    log("Shape of w: {}, shape of U: {}", w.shape, U.shape)
    
    ### Setup data and model
    
    tic = timer()
    
    model = ConvexitySplitting(args.penalty, args.epsilon, args.c_splitting,
                               args.max_iter, args.time_step, args.tolerance)
    
    if args.relative_random_splits is not None:
        split_size = int(np.ceil(args.relative_random_splits * num_samples / num_classes))
    else:
        split_size = args.random_splits
    
    if split_size is None:
        train_mask = np.zeros(num_samples, dtype=bool)
        train_mask[:training_size] = True
        
        u0 = np.zeros(num_samples)
        u0[train_mask] = labels[train_mask]
    
    time_model = timer() - tic
    log('Time for model setup: {:.2f} seconds', time_model)
        
    time_setup = time_distances + time_adjacency + time_eigs + time_model
    log('Setup done in {:.2f} seconds', time_setup)
    
    time_training = []
    accuracies = []
    
    for run in range(args.num_runs):
        aux.set_seed(run)
    
        if split_size is not None:
            train_mask = np.zeros(num_samples, dtype=bool)
            for i in (-1,1):
                label_mask = labels == i
                ind = np.random.choice(range(num_samples), split_size, replace=False,
                                       p=label_mask/label_mask.sum())
                train_mask[ind] = True
            
            u0 = np.zeros(num_samples)
            u0[train_mask] = labels[train_mask]
    
        tic = timer()
        
        u = model.run(u0, w, U)
        
        t = timer() - tic
        time_training.append(t)
    
        prediction = np.sign(u[~train_mask])
        acc = np.mean(prediction == labels[~train_mask])
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
        
        architecture_name = 'AllenCahn'
        if args.rank is not None:
            architecture_name += '_rank{}'.format(args.rank)
        if args.skip_ev != 0:
            architecture_name += '_skip{}'.format(args.skip_ev)
        if args.penalty != 1e10:
            architecture_name += '_pen{:g}'.format(args.penalty)
        if args.epsilon is not None:
            architecture_name += '_eps{:g}'.format(args.epsilon)
        if args.c_splitting is not None:
            architecture_name += '_c{:g}'.format(args.c_splitting)
        if args.time_step != 0.01:
            architecture_name += '_dt{:g}'.format(args.time_step)
        if args.max_iter != 500:
            architecture_name += '_iter{}'.format(args.max_iter)
        if args.tolerance != 1e-8:
            architecture_name += '_tol{:g}'.format(args.tolerance)
        
        
        # if args.sparsify is not None:
        #     architecture_name += '_sparsify{}'.format(args.sparsify)
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
