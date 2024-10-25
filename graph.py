import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_results(filename):
    data = np.load(filename)
    return data['estimates'], data['ground_truth']

def plot_selected_nodes(estimates, ground_truth, time_points, selected_nodes, pdf_filename):
    num_iter = min(estimates.shape[0],80)  # Plot at most 80 iterations
    num_nodes = len(selected_nodes)
    colors = plt.cm.viridis(np.linspace(0, 1, num_nodes))  # Generate colors

    fig, axes = plt.subplots(num_iter, 1, figsize=(8, 4 * num_iter), sharex=True)

    if num_iter == 1:
        axes = [axes]  # If there's only one iteration, put the axes in a list for consistency

    for i, ax in enumerate(axes):
        for j, node in enumerate(selected_nodes):
            # Plot estimates for the selected node
            ax.plot(time_points, estimates[i, :, node-1], linestyle='-', color=colors[j], label=f'n={node}')
            # Plot ground truth for the selected node
            ax.plot(time_points, ground_truth[i, :, node-1], linestyle='--', color=colors[j], label=f'Ground Truth n={node}')
        
        ax.set_title(f'Iteration {i+1}')
        ax.set_xlabel(r'Time $t$')
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_ylabel(r'$\theta(t)$')
        # need smaller font size
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig(pdf_filename, format='pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the trajectory of theta(t) from simulation results for selected nodes.')
    parser.add_argument('--filename', type=str, default ='/Users/moyang/Desktop/topK simulation/theta_est_iter10_time20_n100_Lij200_p0.2_h0.3_lambda0.0001_seed42.npz', help='The filename of the .npz file containing the simulation results.')
    parser.add_argument('--nodes', nargs='+', type=int, help='List of nodes to plot', default=[1,20,40,60,80,100])
    args = parser.parse_args()
    estimates, ground_truth = load_results(args.filename)
    time_points = np.linspace(0, 1, estimates.shape[1])
    pdf = args.filename.replace('.npz', '.pdf')
    plot_selected_nodes(estimates, ground_truth, time_points, args.nodes, pdf)
