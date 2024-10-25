import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import argparse

## The test statistics are all we need
def load_results(filename):
    data = np.load(filename)
    return data['estimates'], data['ground_truth'], data['GMB_W'], data['MSE_error']

directory_path = os.path.expanduser('~/Desktop/Ranking_Revised/PyCode/test_p_150')
file_pattern = f"{directory_path}/*.npz"
npz_files = glob.glob(file_pattern)

output_collect = None

for file_path in npz_files:
    print(file_path)
    estimates, ground_truth, GMB_W, MSE_error = load_results(file_path)
    # Take the MSE
    # Take the mean among (theta_hat - theta_true) ** 2
    output = np.mean(MSE_error, axis = (1,2))
    if output_collect is None:
        output_collect = output
    else:
        output_collect = np.append(output_collect, output, axis = 0)