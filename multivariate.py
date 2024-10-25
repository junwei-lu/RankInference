import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from tqdm import tqdm

'''
Now we consider multivariate
t is now a d-dimensional vector
dim is the number of dimensions: len(t) = dim
'''

def theta_t(t, n):
    # Generate theta(t)
    # Let dim be some number irrelevant to the structure
    # Take summations for all t[i], for example
    # theta = np.exp(0.01 * t * np.arange(1, n + 1))
    theta = np.exp(sum([0.01 * t[i] * np.arange(1, n + 1) for i in range(len(t))]))
    theta -= np.mean(theta)
    return theta


'''
K is the kernel function. It takes two arguments: u and h.
u is the difference between the time of the comparison and the time of interest.
h is the bandwidth of the kernel.
The function returns the value of the kernel function at u, normalized by h.
'''

# Here u is a vector with dimension dim;
# Here we consider a formality of multiplication

def K(u, h):
    uh = u / h
    return np.prod(0.75 * (1 - uh ** 2) * (np.abs(uh) <= 1) / h)

'''
generate_data generates synthetic data for the model.
L_ij is the number of comparisons for each pair of entities.
p is the probability of edge existence.
n is the number of entities.
The function returns a dictionary with the following keys:
- edge: Adjacency matrix of the graph.
- Y_l: Binary outcomes of the comparisons.
- time: Time points of the comparisons.
'''


def generate_data(L_ij, p, n, dim):
    L = max(L_ij)
    edge = np.full((n, n), np.nan)
    Y_l = np.full((n * n, L), np.nan)
    time = np.full((n * n, L, dim), np.nan)

    for i in range(n - 1):  # Adjusted for Python's 0-indexing
        for j in range(i + 1, n):
            edge[i, j] = np.random.binomial(1, p)
            edge[j, i] = edge[i, j]
            if edge[i, j] == 1:
                # Uniform distribution for timepoint selection
                time[n * i + j, :L_ij[n * i + j], ] = np.random.uniform(0, 1, size = (L_ij[n * i + j], dim))
                time[n * j + i, :L_ij[n * i + j], ] = time[n * i + j, :L_ij[n * i + j], ]
                # Assuming theta_t function is defined elsewhere and provided
                theta = [theta_t(t, n) for t in time[n * i + j, :L_ij[n * i + j]]]
                for k in range(L_ij[n * i + j]):
                    wj = np.exp(theta[k][j])
                    wi = np.exp(theta[k][i])
                    Y_l[n * i + j, k] = np.random.binomial(1, wj / (wi + wj))  # j wins
                    # i wins calculation
                    Y_l[n * j + i, :L_ij[n * i + j]] = 1 - Y_l[n * i + j, :L_ij[n * i + j]]

    return {'edge': edge, 'Y_l': Y_l, 'time': time}


'''
L_t2 is the log-likelihood function for the model.
Y_l is the binary outcomes of the comparisons.
time is the time points of the comparisons.
edge is the adjacency matrix of the graph.
L_ij is the number of comparisons for each pair of entities.
theta is the vector of theta values.
t is the time of interest.
h is the bandwidth of the kernel.
lambda_val is the L2 penalization parameter.
n is the number of entities.
p is the probability of edge existence.
The function returns the value of the log-likelihood function at theta.
'''

## h is now 1D array with len(h) = dim
def L_t2(Y_l, time, edge, L_ij, theta, t, h, lambda_val, n, p):
    M = 0
    Z = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if edge[i, j] == 1:
                for k in range(L_ij[n * i + j]):
                    time_diff = time[n * i + j, k] - t
                    kernel_val = K(time_diff, h)
                    Z += kernel_val
                    M += kernel_val * (
                            -Y_l[n * i + j, k] * (theta[j] - theta[i]) +
                            np.log(1 + np.exp(theta[j] - theta[i]))
                    )
    regularization_term = lambda_val / 2 * np.sum(theta ** 2)
    return M / Z + regularization_term


'''
gradient2 is the gradient function for the model.
Y_l is the binary outcomes of the comparisons.
time is the time points of the comparisons.
edge is the adjacency matrix of the graph.
L_ij is the number of comparisons for each pair of entities.
theta is the vector of theta values.
t is the time of interest.
h is the bandwidth of the kernel.
lambda_val is the L2 penalization parameter.
n is the number of entities.
p is the probability of edge existence.
The function returns a dictionary with the following keys:
- grad1: The gradient of the log-likelihood function at theta.
- grad2: The Hessian of the log-likelihood function at theta.
'''


def gradient2(Y_l, time, edge, L_ij, theta, t, h, lambda_val, n, p):
    # Initialize grad1 as a zero vector of length n
    grad1 = np.zeros(n)
    # Initialize grad2 as a zero matrix of size n x n
    grad2 = np.zeros((n, n))
    # E is the identity matrix of size n x n
    E = np.eye(n)
    # Calculate Z, which is used in the normalization of grad1 and grad2
    Z = 0

    # Iterate over all (i, j) pairs
    for i in range(1, n):  # Adjusted for Python's 0-indexing
        for j in range(i):
            if edge[j, i] == 1:
                # Calculate the kernel values for all comparisons between entities i and j
                # Generate this to multivariate
                # K_values = K(time[n * i + j - n, :L_ij[n * i + j - n]] - t, h)
                time_diff_mat = time[n * i + j, :L_ij[n * i + j]] - t
                K_values = np.array([K(TD, h) for TD in time_diff_mat])
                # Update the denominators
                Z_update = np.sum(K_values)
                # Update grad1 based on the formula provided in the document
                grad1_update = np.sum(K_values * (-Y_l[n * i + j, :L_ij[n * i + j]] +
                                                  np.exp(theta[j]) / (np.exp(theta[i]) + np.exp(theta[j])))) * (
                                           E[j] - E[i])
                Z += Z_update
                grad1 += grad1_update

                # Update grad2 based on the formula provided in the document
                grad2_update = np.sum(K_values) * np.exp(theta[i] + theta[j]) / (
                            np.exp(theta[i]) + np.exp(theta[j])) ** 2 * \
                               np.outer((E[i] - E[j]), (E[i] - E[j]).T)
                grad2 += grad2_update

    # Normalize grad1 and grad2 by Z and add regularization term to grad1
    grad1 = grad1 / Z + lambda_val * theta
    grad2 = grad2 / Z

    return {'grad1': grad1, 'grad2': grad2}


'''
f_theta_h2 is the function that estimates the trajectory of theta(t).
Y_l is the binary outcomes of the comparisons.
time is the time points of the comparisons.
edge is the adjacency matrix of the graph.
L_ij is the number of comparisons for each pair of entities.
t is the time of interest.
lambda_val is the L2 penalization parameter.
h is the bandwidth of the kernel.
n is the number of entities.
p is the probability of edge existence.
The function returns the estimated trajectory of theta(t).
'''


def f_theta_h2(Y_l, time, edge, L_ij, t, lambda_val, h, n, p, a=0.1, b=0.1):
    theta_h = np.zeros(n)  # Assuming 'n' is defined elsewhere as the number of entities
    grad1 = gradient2(Y_l, time, edge, L_ij, theta_h, t, h, lambda_val, n, p)['grad1']
    sum_iter = 0
    ## This threshold is tunable
    ## For multivariate, this is likely all we could do
    ## Reduce the lim of iters; let it go!
    while np.sum(np.abs(grad1)) > 1e-6 and sum_iter <= 50:
    ## while np.sum(grad1 ** 2) > 1e-6 and sum_iter <= 50:
        m = 30
        sum_iter += 1
        reduce = 0
        while L_t2(Y_l, time, edge, L_ij, theta_h - m * grad1, t, h, lambda_val, n, p) > \
                L_t2(Y_l, time, edge, L_ij, theta_h, t, h, lambda_val, n, p) - a * m * np.dot(grad1, grad1):
            m = b * m
            reduce += 1
        theta_h = theta_h - m * grad1
        grad1 = gradient2(Y_l, time, edge, L_ij, theta_h, t, h, lambda_val, n, p)['grad1']
        if reduce >= 2:
            break
    ## print("sum_iter:", sum_iter)
    ## print("reduce:", reduce)
    return theta_h

## Generate Statistic W
## T is the time grid
def generate_W(Y_l, time, edge, t, L_ij, h, n, p, w_iter, xi, theta_h, dim):
    ## Suppose that t is the anchored parameter: t is an dim-dimensional iterable
    ## Replace m in the original code to be translated
    ## t is the real time, theta_h = theta_h(t) is the estimated value for plug-in

    ## Initialization
    Vm = np.zeros((n, w_iter))
    Gm = np.zeros((n, w_iter))

    ## iter_1 for m
    ## iter_2 for k
    for iter_1 in range(n):
        for iter_2 in range(n):
            if  iter_1 != iter_2 and edge[iter_1, iter_2] == 1:
                K_values = K(time[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1]] - t, h)
                ## Should be the same among groups
                Vm[iter_1,] += np.sum(K_values * (np.exp(theta_h[iter_1] - theta_h[iter_2]) /
                                                  (1 + np.exp(theta_h[iter_1] - theta_h[iter_2])) ** 2))
                for iter_samples in range(w_iter):
                        Gm[iter_1, iter_samples] += (K_values * xi[n * iter_2 + iter_1, : ,iter_samples]) @ \
                                            (-Y_l[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1]] +
                                             np.exp(theta_h[iter_1]) / (np.exp(theta_h[iter_1]) + np.exp(theta_h[iter_2])))

    W = ((np.sqrt(h)) ** dim) * np.divide(Gm, Vm, where = Vm != 0, out = np.zeros_like(Gm))
    ## W = - np.sqrt(n * p * max(L_ij) * h) * (Gm / Vm)
    return W


# Now `estimates` contains the theta estimates for each simulation
# Subject to edits in multivariate
def plot_trajectory(estimates, true_theta, t):
    """
    Plots the estimated trajectories.

    Parameters:
    - estimates: Array of estimated theta values.
    - t: Time points.
    - true_theta: Array of true theta values for comparison (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, estimates[:, 40], label=f'Estimate {1} for node 40')

    if true_theta is not None:
        plt.plot(t, true_theta[:, 40], 'k--', label='True Theta')

    plt.xlabel('Time')
    plt.ylabel('Theta')
    plt.title('Trajectory Estimates')
    plt.legend()
    plt.show()


def parse_argument():
    parser = argparse.ArgumentParser(description='Estimate the trajectory of theta(t)')
    parser.add_argument('--h', type=float, default=0.3, help='Kernel bandwidth')
    parser.add_argument('--time_num', type=int, default=11, help='Number of time points')
    parser.add_argument('--lambda_val', type=float, default=0.00001, help='L2 penalization')
    parser.add_argument('--n', type=int, default=100, help='Number of entities')
    parser.add_argument('--p', type=float, default=0.2, help='Probability for edge existence')
    parser.add_argument('--L_val', type=int, default=400, help='Number of comparisons for each pair of entities')
    parser.add_argument('--iter_num', type=int, default=50, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', type=int, default=8, help='Number of CPUs')
    # Add the default number dim
    parser.add_argument('--dim', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--w_iter', type=int, default=100, help='Samples for Iteration for Confidence Band')
    args = parser.parse_args()
    print(args)
    return args

## Make this a function of dim as well
## Supposedly, t is now a dim-dimensional grid
def compute_theta_h(i, t, L_ij, p, n, lambda_val, h, seed, w_iter, dim):
    np.random.seed(i * seed)
    ## Gaussian Multiplier Bootstrapping
    shape = (n * n, max(L_ij), w_iter)
    xi = np.random.normal(loc=0, scale=1, size=shape)
    data = generate_data(L_ij, p, n, dim)  # Generate synthetic data
    edge = data['edge']
    Y_l = data['Y_l']
    time = data['time']
    # Estimation for theta(t)
    theta_h = f_theta_h2(Y_l, time, edge, L_ij, t, lambda_val, h, n, p)
    W = generate_W(Y_l, time, edge, t, L_ij, h, n, p, w_iter, xi, theta_h, dim)
    ground_truth = theta_t(t, n)  # Generate ground truth
    return i, t, theta_h, ground_truth, W

## Generate multivariate grids
def generate_grid(begin, end, dim, time_num):
    # Generate n_points evenly spaced values between 0 and 1 for each dimension
    axis_values = np.linspace(begin, end, time_num)

    # Initialize an empty list to store the grid points
    grid_points = []

    # Generate grid points
    def generate_point(coord=[]):
        if len(coord) == dim:
            grid_points.append(coord)
            return
        for value in axis_values:
            generate_point(coord + [value])

    # Start generating grid points
    generate_point()
    return np.array(grid_points)


if __name__ == "__main__":
    args = parse_argument()
    h = args.h
    time_num = args.time_num
    lambda_val = args.lambda_val
    n = args.n
    L_val = args.L_val
    p = args.p
    iter_num = args.iter_num
    seed = args.seed
    w_iter = args.w_iter
    dim = args.dim

    ## Let's try this for multivariate
    ## T = np.linspace(0, 1, time_num)
    T = generate_grid(0, 1, dim, time_num)
    ## I doubt if this is the real coef here
    ## This could be a typo
    L_ij = np.full((n * n,), L_val)

    all_combinations = [(i, t, L_ij, p, n, lambda_val, h, seed, w_iter, dim) for i in range(iter_num) for t in T]

    # Prepare for parallel processing
    with ProcessPoolExecutor(max_workers=args.cpu) as executor:
        # Submit all the tasks and create a list of futures
        futures = [executor.submit(compute_theta_h, *comb) for comb in all_combinations]

        # Wrap the as_completed iterator with tqdm for a progress bar
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task"):
            res = future.result()
            results.append(res)
    # Organize results into a structured format
    estimates = np.zeros((iter_num, time_num, n))
    ground_truth = np.zeros((iter_num, time_num, n))
    # W is n-by-w_iter matrix among i in iter_num and t in time_num
    GMB_W = np.zeros((iter_num, time_num, n, w_iter))

    for i, t, theta_h, true_theta, W in results:
        estimates[i, np.where(T == t)[0][0]] = theta_h
        ground_truth[i, np.where(T == t)[0][0]] = true_theta
        GMB_W[i, np.where(T == t)[0][0]] = W

    result_dict = {'estimates': estimates, 'ground_truth': ground_truth, 'GMB_W': GMB_W}
    np.savez(f'iter{iter_num}_n{n}_Lij{L_val}_p{p}_h{h}_lambda{lambda_val}_dim{dim}.npz', **result_dict)

