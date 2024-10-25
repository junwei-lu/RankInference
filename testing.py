import numpy as np
import matplotlib.pyplot as plt
import argparse

## The test statistics are all we need
def load_results(filename):
    data = np.load(filename)
    return data['estimates'], data['ground_truth'], data['GMB_W'], data['MSE_error']

## Recall the GMB_W data formality:
## (iter_num, time_num, n, iter_w)
## Fix iter_num and iter_w, we get the 2D array for the W statistics to play around.
## alpha is the quantile 1-alpha percent

def sup_statistics(W, alpha):
    ## Set up the statistics
    ## For one indicator of different bootstrapping, we consider
    max_statistic = np.full(W.shape[0], np.nan)
    for iter_0 in range(W.shape[0]):
        ## For each iter_w, we create a slot to save its test statistics
        TS = np.full(W.shape[3], np.nan)
        ## Could be made more efficient, but this is less prone to mistakes
        for iter_1 in range(W.shape[3]):
            TS[iter_1] = np.max(W[iter_0, : , : ,iter_1])
        ## Calculate the quantiles of TS
        max_statistic[iter_0] = np.quantile(TS, 1 - alpha / 2)
        ## max_statistic[iter_0] = np.quantile(TS, 1 - alpha/2)
    return max_statistic

## True +- multiplier * max_statistic should work for each expected trajectory
## Just need to think about ways to draw a bowtie plot for the specific cases


## For pairwise statistics: the comparison is conducted among all available pairs on W_i - W_j
## This requires some tricks to make the things done
## Could be optimized for efficiency but once we get the results from main.py then it should be fine

def pairwise_statistics(W, alpha):
    max_pairwise = np.full(W.shape[0], np.nan)
    ## Now we should consider two iterables
    ## Still save all the sampled max statistics
    for iter_0 in range(W.shape[0]):
        ## slot for TS among all iterations
        T_ij = np.full(W.shape[3], np.nan)
        for iter_1 in range(W.shape[3]):
            TS = np.full(int(W.shape[3] * (W.shape[3] - 1) / 2), np.nan)
            sum = 0
            for iter_i in range(W.shape[2] - 1):
                for iter_j in range(iter_i + 1, W.shape[2]):
                    TS[sum] = np.max(W[iter_0, : , iter_i ,iter_1] - W[iter_0, : , iter_j ,iter_1])
                    sum += 1
            T_ij[iter_1] = np.max(TS)
        max_pairwise[iter_0] = np.quantile(T_ij, 1 - alpha / 2)
    return max_pairwise


## Finally, we get the Top-K analysis and doing this should be good for our output.
## Top K is a bit tricky since we always need to anchor the one ranking on the K-th position.
## The max_pairwise TS is helpful, but we need to specify a more ideal way to use this.

## index_i specifies the index of item that we use to anchor the top-K-th item for score theta
## K stands for the specific top-K that is anchored at each evaluated time_num
## Once we anchor the Kth item for each evaluated point, we consider

def find_top_K(estimates, index_i, K):
    estim_K = np.full(estimates.shape[1], np.nan)
    ## The Top-K task Conf Band TS
    values = estimates[index_i,: , : ]
    for iter_0 in range(values.shape[0]):
        estim_K[iter_0] = np.partition(values[iter_0, :], -K)[-K]
    return estim_K

## The rest task is to design the plots. All the TS and theta_hat to be evaluated are ready here.
## We should write the code to make it runnable for a more generalizable setup
## The images could be tuned

def plot_selected_nodes(estimates, ground_truth, time_points, selected_nodes, pdf_filename):
    num_iter = min(estimates.shape[0], 80)  # Plot at most 80 iterations
    num_nodes = len(selected_nodes)
    colors = plt.cm.viridis(np.linspace(0, 1, num_nodes))  # Generate colors

    fig, axes = plt.subplots(num_iter, 1, figsize=(8, 4 * num_iter), sharex=True)

    if num_iter == 1:
        axes = [axes]  # If there's only one iteration, put the axes in a list for consistency

    for i, ax in enumerate(axes):
        for j, node in enumerate(selected_nodes):
            # Plot estimates for the selected node
            ax.plot(time_points, estimates[i, :, node - 1], linestyle='-', color=colors[j], label=f'n={node}')
            # Plot ground truth for the selected node
            ax.plot(time_points, ground_truth[i, :, node - 1], linestyle='--', color=colors[j],
                    label=f'Ground Truth n={node}')

        ax.set_title(f'Iteration {i + 1}')
        ax.set_xlabel(r'1D Prompt $x$')
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_ylabel(r'$\theta(x)$')
        # need smaller font size
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig(pdf_filename, format='pdf')


def plot_selected_band(estimates, ground_truth, time_points, selected_nodes, pdf_filename, bandwidth):
    num_iter = min(estimates.shape[0], 80)  # Plot at most 80 iterations
    num_nodes = len(selected_nodes)
    colors = plt.cm.viridis(np.linspace(0, 1, num_nodes))  # Generate colors

    fig, axes = plt.subplots(num_iter, 1, figsize=(8, 4 * num_iter), sharex=True)

    if num_iter == 1:
        axes = [axes]  # If there's only one iteration, put the axes in a list for consistency

    for i, ax in enumerate(axes):
        for j, node in enumerate(selected_nodes):
            # Plot estimates for the selected node
            ax.plot(time_points, estimates[i, :, node - 1], linestyle='-', color=colors[j], label=f'n={node}')
            ax.fill_between(time_points, estimates[i, :, node - 1] - bandwidth, estimates[i, :, node - 1] + bandwidth, color=colors[j], alpha=0.5, label=f'Confidence Band for n={node}')
            # Plot ground truth for the selected node
            ax.plot(time_points, ground_truth[i, :, node - 1], linestyle='--', color=colors[j],
                    label=f'Ground Truth n={node}')


        ax.set_title(f'Iteration {i + 1}')
        ax.set_xlabel(r'Prompt $x$')
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_ylabel(r'$\theta(x)$')
        # need smaller font size
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig(pdf_filename, format='pdf')


## Need a function to give the coverage results of the conf band
## Should be able to work on the conf percentage, or make this usable for further processing
## Let's take bandwidth as a vector
## Truncation means the item numbers on the edge such that should not be considered in the conf band research
## For 2D, this need to be refined; the current once is working so do not change so far
## Should add into  input

def coverage_mat(estimates, ground_truth, bandwidth, truncation):
    entrywise_result = np.logical_and(ground_truth <= estimates + bandwidth, ground_truth >= estimates - bandwidth)
    coverage_output = entrywise_result.astype(int)
    cover_mat = np.min(coverage_output[:, truncation:estimates.shape[1]-truncation ,:], axis = 1)
    return (cover_mat)

## And we also need to arrange this coverage matrix and calculate a coverage for some specific entry
## In our formality, the coverage matrix contains 1s and 0s
## In each of such matrices called cover_mat, we have to consider whether they cover a pairwise test among itself
## Say, among the 50 samples, it is testable whether the coverage is achieved.


## Top K is a bit tricky
## Say

## Need to consider some brief techniques for matplotlib;
## We use the following code for testing:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the trajectory of theta(t) from simulation results for selected nodes.')
    parser.add_argument('--filename', type=str, default ='test.npz', help='The filename of the .npz file containing the simulation results.')
    parser.add_argument('--nodes', nargs='+', type=int, help='List of nodes to plot', default=[1,25,50,75,100])
    parser.add_argument('--alpha', type=float, default = 0.05, help='The significance value of the test.')
    parser.add_argument('--index_i', nargs='+', type=int, default = [1], help='The index of selected initialization to create the plots.')
    parser.add_argument('--K', type=int, default = 1, help='The index K for Top-K tests.')
    args = parser.parse_args()
    estimates, ground_truth, GMB_W, MSE_error = load_results(args.filename)
    ## First, we need the test statistics for the bowtie plot
    ## For time points, we need to specify this for 3D setup
    time_points = np.linspace(0, 1, estimates.shape[1])
    ## Selected nodes: stand for the nodes to be proposed here; args.nodes
    pdf = args.filename.replace('.npz', '.pdf')
    plot_selected_nodes(estimates, ground_truth, time_points, args.nodes, pdf)
    ## plot_selected_nodes(estimates, ground_truth, time_points, nodes, pdf)
    pdf_band = args.filename.replace('.npz', '_band.pdf')
    bandwidth = np.mean(sup_statistics(GMB_W, alpha))
    # pdf_band = ('band.pdf')
    plot_selected_band(estimates, ground_truth, time_points, [1, 25, 50, 75, 100], pdf_band, bandwidth)