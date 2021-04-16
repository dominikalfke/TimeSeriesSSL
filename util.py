
import numpy as np
import os
import warnings

# Values copied from seeds.py in github.com/klicperajo/gdc
seeds = [
    2406525885, 3164031153, 1454191016, 1583215992, 765984986,
    258270452, 3808600642, 292690791, 2492579272, 1660347731,
    902096533, 1295255868, 3887601419, 2250799892, 4099160157,
    658822373, 1105377040, 1822472846, 2360402805, 2355749367,
    2291281609, 1241963358, 3431144533, 623424053, 78533721,
    1819244826, 1368272433, 555336705, 1979924085, 1064200250,
    256355991, 125892661, 4214462414, 2173868563, 629150633,
    525931699, 3859280724, 1633334170, 1881852583, 2776477614,
    1576005390, 2488832372, 2518362830, 2535216825, 333285849,
    109709634, 2287562222, 3519650116, 3997158861, 3939456016,
    4049817465, 2056937834, 4198936517, 1928038128, 897197605,
    3241375559, 3379824712, 3094687001, 80894711, 1598990667,
    2733558549, 2514977904, 3551930474, 2501047343, 2838870928,
    2323804206, 2609476842, 1941488137, 1647800118, 1544748364,
    983997847, 1907884813, 1261931583, 4094088262, 536998751,
    3788863109, 4023022221, 3116173213, 4019585660, 3278901850,
    3321752075, 2108550661, 2354669019, 3317723962, 1915553117,
    1464389813, 1648766618, 3423813613, 1338906396, 629014539,
    3330934799, 3295065306, 3212139042, 3653474276, 1078114430,
    2424918363, 3316305951, 2059234307, 1805510917, 1327514671
]


def set_seed(index, with_torch=False):
    r"""Pick the seed with the given index from the seeds list and use it to
    seed the random number generators of numpy and torch. If the index exceeds
    the number of available seeds, nothing is done.The seed list was copied 
    from the file seeds.py in github.com/klicperajo/gdc"""
    if index < len(seeds):
        np.random.seed(seeds[index])
        if with_torch:
            import torch
            torch.manual_seed(seeds[index])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif index == len(seeds):
        warnings.warn('Number of available seeds exceeded. Future runs will not be reproducible.')
        

def print_results(accuracies, setup_times, training_times, *dictionaries, 
                      architecture=None, dataset=None, print_all=False, **kwargs):
    r"""Print a few lines summarizing the results of a named architecture on a
    named dataset. If additionally dictionaries and/or keyword arguments are
    given, all their key-value pairs are also printed. If print_all is True,
    the individual results of all runs are also printed.
    """
    
    def p(varname, var, unit=''):
        if var is not None:
            if np.size(var) > 1:
                print(" - Average {}: {} {} +- {}".format(varname, np.mean(var), unit, np.std(var)))
            else:
                print(" - {}: {} {}".format(varname, np.mean(var), unit))
                
    N = len(accuracies)
        
    if architecture is not None:
        print("Summary of {} runs with architecture \"{}\"".format(N, architecture)
              + (":" if dataset is None else " on dataset \"{}\":".format(dataset)))
    
    if N == 0:
        return
    
    p("Accuracy", 100*np.array(accuracies), '%')
    p("Setup Time", setup_times, 's')
    p("Training Time", training_times, 's')
    if setup_times is not None and training_times is not None:
        p("Total Time", np.add(setup_times, training_times), 's')
    
    if len(dictionaries) > 0 or len(kwargs) > 0:
        print()
        for d in [*dictionaries, kwargs]:
            if not isinstance(d, dict):
                d = d.__dict__
            for key, val in d.items():
                print(' - {} = {}'.format(key, val))
    
    if N > 1 and print_all:
        print()
        print("Individual run results:")
        if isinstance(setup_times, list):
            print(" Run   SetupTime  TrainingTime  Accuracy")
            for run in range(N):
                print(" {: 3d}  {: 12.4f}  {: 12.4f}  {:.4f}".format(run+1, setup_times[run], training_times[run], 100*accuracies[run]))
        else:
            print(" Run  TrainingTime  Accuracy")
            for run in range(N):
                print(" {: 3d}  {: 12.4f}  {:.4f}".format(run+1, training_times[run], 100*accuracies[run]))
            

def save_results(dir, dataset, architecture, *args, **kwargs):
    r"""Save the output of print_results in a file. The .TXT file is named 
    after the architecture and created in a directory named after the dataset 
    within the given parent directory."""
    from contextlib import redirect_stdout

    dir = os.path.join(dir, dataset)
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, architecture + '.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            print_results(*args, architecture=architecture, dataset=dataset, print_all=True, **kwargs)
    print('Results saved to file {}'.format(filename))