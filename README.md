# HSG-12M
HSG-12M Spatial Multigraph Dataset.

This repository contains the generation code for the HSG-12M dataset, the processing code to derive the six dataset variants, the preliminary featurization and processing to PyTorch Geometric Dataset (both in memory and on disk), and the benchmarking code used in the companion paper "HSG-12M: A Large-Scale Spatial Multigraph Dataset".

The 1401 data files are publicly available at [Dataverse](https://doi.org/10.7910/DVN/PYDSSQ).

## Installation
```bash
$ conda create -n hsg python=3.12 # python>=3.11
$ conda activate hsg
$ export CUDA=cu124

$ git clone https://github.com/sarinstein-yan/HSG-12M.git
$ cd HSG-12M
$ bash scripts/init_env.sh
$ pip install -e .
```

## Dataset Loading
```python
from hsg.pyg import HSGOnDisk, HSGInMemory
```