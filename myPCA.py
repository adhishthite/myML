# Implement PCA

import numpy as np
from numpy import linalg as LA
from numpy import genfromtxt as GFT
import matplotlib.pyplot as plt


# MAIN FUNCTION
def main():
    # Load Data
    input_data = load_data()

    # Implement PCA
    pca_results = do_pca(input_data)

    # Plot Data
    plot_data(pca_results)

    exit()


# LOAD THE INPUT DATA
def load_data():
    # Original Dataset
    input_data = GFT('datasets/linear_regression_test_data.csv', skip_header=1, delimiter=',')

    # Delete first column
    input_data = np.delete(input_data, 0, axis=1)

    return input_data


# PERFORM PCA ON INPUT DATA
def do_pca(input_data):
    # As we are performing PCA on 'x' and 'y' axis, we remove the 'y_theoretical' value
    data = input_data[:, :2]

    # Calculate mean centered data matrix from given data

    column_mean = data.mean(axis=0)
    column_mean_all = np.tile(column_mean, reps=(data.shape[0], 1))
    data_mean_centered = data - column_mean_all

    # Get covariance matrix
    covariance_matrix = np.cov(data_mean_centered, rowvar=False)


    # Calculate the Eigenvectors and Eigenvalues of the Covariance Matrix
    eigen_values, eigen_vectors = LA.eig(covariance_matrix)
    indices = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[indices]
    eigen_vectors = eigen_vectors[:, indices]

    # Calculate PCA - data * eigenvectors
    pca_scores = np.matmul(data_mean_centered, eigen_vectors)

    # Collect PCA Results in a JSON file so that we can the results to plot graphs
    pca_results = {'input_data': input_data,
                   'data': data,
                   'mean_centered_data': data_mean_centered,
                   'variance': eigen_values,
                   'loadings': eigen_vectors,
                   'scores': pca_scores}

    return pca_results


# PLOT DATA ON GRAPH
def plot_data(pca_results):
    # Plotting the Original Data

    scale_factor = 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    plt.legend(loc='upper left')
    plt.suptitle('Principal Component Analysis')

    # y vs x
    ax.scatter(pca_results['input_data'][:, 0], pca_results['input_data'][:, 1], color='blue',
               label='Input Data : \'y\' vs \'x\'')

    # y_theoretical vs x
    ax.scatter(pca_results['input_data'][:, 0], pca_results['input_data'][:, 2], color='red',
               label='Input Data : \'y_theoretical\' vs \'x\'')

    # PC1
    ax.plot([0, (-1) * scale_factor * pca_results['loadings'][0, 0]], [0, (-1) * scale_factor * pca_results['loadings'][1, 0]],
            color='green', linewidth=5, label='Principal Component 1')

    plt.legend(loc='upper left')

    plt.savefig('plots/PCA.jpg')

    plt.close()


if __name__ == '__main__':
    main()
