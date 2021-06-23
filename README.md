# TimeSeriesSSL
A study of distance measures and learning methods for semi-supervised learning on time series data

## Prerequesites

Other than scipy/numpy, the following distance measures and learning methods have their own prerequesites:
* DTW distance: `dtaidistance` from [https://github.com/wannesm/dtaidistance]
* SoftDTW distance: `sdtw` from [https://github.com/mblondel/soft-dtw]
* Matrix Profile distance: `stumpy` from [https://github.com/TDAmeritrade/stumpy]
* Graph Convolutional Network: `torch` and `torch-geometric` from [https://github.com/rusty1s/pytorch_geometric]

## Usage

Each of the four methods has its own Python script `run-method.py` which is controlled by command-line options. Run, e.g., `python run-gcn.py --help` for a list of arguments.
