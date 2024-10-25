# Rank Inference for Domain Knowledge

[![GitHub](https://img.shields.io/badge/github-junwei--lu/RankInference-blue)](https://github.com/junwei-lu/RankInference)

A Python package for performing ranking inference on time-varying paired comparison data. This package implements methods for both univariate and multivariate time-varying ranking estimation with statistical inference capabilities.

## Overview

RankInference is designed to analyze paired comparison data that evolves over time. It provides tools for:

- Estimating time-varying rankings from paired comparison outcomes
- Handling both univariate and multivariate time settings
- Generating confidence bands using Gaussian Multiplier Bootstrap
- Parallel processing support for computational efficiency
- Visualization of ranking trajectories

The package uses kernel-based methods for smooth estimation of ranking trajectories and implements L2 regularization for stability.

<img src="https://github.com/junwei-lu/RankInference/blob/main/img/rankinference.png"/>


## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- tqdm
- concurrent.futures (included in Python standard library)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/junwei-lu/RankInference.git
cd RankInference
```

2. Install required packages:
```bash
pip install numpy matplotlib tqdm
```

## Usage Examples

### Basic Usage - Univariate Analysis

```python
import numpy as np
from main import parse_argument, compute_theta_h

# Set up parameters
args = parse_argument()
n = 100  # number of entities
L_val = 400  # comparisons per pair
p = 0.2  # edge probability
h = 0.3  # kernel bandwidth
lambda_val = 0.00001  # L2 regularization parameter

# Generate time points
T = np.linspace(0, 1, 21)
L_ij = np.full((n * n,), L_val)

# Run estimation for a single time point
i = 0  # iteration index
t = 0.5  # time point
result = compute_theta_h(i, t, L_ij, p, n, lambda_val, h, seed=42, w_iter=100)
```

### Multivariate Analysis

```python
from multivariate import parse_argument, compute_theta_h, generate_grid

# Set up parameters
args = parse_argument()
dim = 2  # number of dimensions
time_num = 11  # points per dimension
n = 100  # number of entities

# Generate multidimensional grid
T = generate_grid(0, 1, dim, time_num)

# Run estimation for a single grid point
t = T[0]  # first grid point
result = compute_theta_h(i=0, t=t, L_ij=L_ij, p=p, n=n, 
                        lambda_val=lambda_val, h=h, seed=42, 
                        w_iter=100, dim=dim)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Setup parallel processing parameters
iter_num = 50
cpu_count = 8

# Create parameter combinations
all_combinations = [(i, t, L_ij, p, n, lambda_val, h, seed, w_iter) 
                   for i in range(iter_num) 
                   for t in T]

# Run parallel processing
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(compute_theta_h, *comb) 
              for comb in all_combinations]
    results = []
    for future in tqdm(as_completed(futures), 
                      total=len(futures), 
                      desc="Processing"):
        results.append(future.result())
```

### Command Line Interface

Both univariate and multivariate analyses can be run from the command line:

```bash
# Univariate analysis
python main.py --h 0.3 --time_num 21 --lambda_val 0.00001 --n 100 --p 0.2 --L_val 400 --iter_num 50 --cpu 8

# Multivariate analysis
python multivariate.py --h 0.3 --time_num 11 --lambda_val 0.00001 --n 100 --p 0.2 --L_val 400 --iter_num 50 --cpu 8 --dim 2
```

### Parameter Descriptions

- `h`: Kernel bandwidth for smoothing
- `time_num`: Number of time points to evaluate
- `lambda_val`: L2 regularization parameter
- `n`: Number of entities to rank
- `p`: Probability of edge existence
- `L_val`: Number of comparisons per pair
- `iter_num`: Number of iterations for bootstrap
- `cpu`: Number of CPU cores for parallel processing
- `dim`: Number of dimensions (multivariate only)
- `w_iter`: Number of bootstrap samples

### Output

The package saves results in NPZ files containing:
- Estimated rankings
- Ground truth values
- Bootstrap statistics

Results are saved with filenames encoding the parameters:
```
iter{iter_num}_n{n}_Lij{L_val}_p{p}_h{h}_lambda{lambda_val}.npz
```

For multivariate analysis:
```
iter{iter_num}_n{n}_Lij{L_val}_p{p}_h{h}_lambda{lambda_val}_dim{dim}.npz
```
