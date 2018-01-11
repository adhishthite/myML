# Implement LDA

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Load the Dataset
def load_data():
    in_file_name = "datasets/SCLC_study_output_filtered_2.csv"
    data_in = pd.read_csv(in_file_name, index_col=0)

    X = data_in.as_matrix()

    y = np.concatenate((np.zeros(20), np.ones(20)))

    return X, y


# Do LDA
def do_LDA(X, y, class_count):

    X_group = {}
    u = {}
    n = X.shape[1]

    # Calculate Total Input Mean
    total_mean = np.mean(X, axis=0)

    for i in range(class_count):
        indices = np.where(y == i)[0]
        X_group[i] = X[indices]

        u[i] = np.mean(X_group[i], axis=0)

    # SCATTER MATRIX CALCULATION

    # Scatter Within

    S = {}
    scatter_within = np.zeros((n,n))
    for i in range (class_count):
        S[i] = np.zeros((n, n))
        for j in range(20):
            difference_mean = X_group[i][j, :] - u[i]
            difference_mean = np.array(difference_mean)
            difference_mean = difference_mean.reshape((n, 1))
            S[i] = S[i] + difference_mean.dot(difference_mean.T)

        scatter_within += S[i]

    scatter_within_inverse = np.linalg.inv(scatter_within)

    # Scatter Between

    # Create a 2-D Mean Vector Class
    mean_vector_class = np.array([u[0], u[1]])

    scatter_between = np.zeros((n,n))

    for i, mean_vector in enumerate(mean_vector_class):
        mean_vector = mean_vector.reshape(n, 1)
        total_mean = total_mean.reshape(n, 1)
        scatter_between = np.add(scatter_between, (20 * (mean_vector - total_mean).dot((mean_vector - total_mean).T)))

    # Calculate Eigenvalues and Eigenvectors

    e_vals, e_vecs = np.linalg.eig(np.dot(scatter_within_inverse, scatter_between))

    # Sort in descending order
    index = e_vals.argsort()[::-1]
    e_vals = e_vals[index]
    e_vecs = e_vecs[:, index]

    # Calculating output
    output = X.dot(e_vecs[:, 0])

    return output


def do_sklearnLDA(X, y, class_count):
    sklearn_LDA = LDA(n_components=class_count)
    sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)

    return sklearn_LDA_projection


def plot_graph(output, title):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlabel(r'$W_1$')
    ax.set_ylabel('')
    ax.plot(output[:20], np.zeros(20), linestyle='None', marker='o', markersize='6', color='blue', label='NSCLC')
    ax.plot(output[20:], np.zeros(20), linestyle='None', marker='x', markersize='6', color='red', label='SCLC')

    ax.legend()

    plt.savefig("plots/LDA - " + title, dpi=500)
    plt.close()


# Main
def main():

    X, y = load_data()

    # Using own code to implement LDA
    output = do_LDA(X, y, class_count=2)
    plot_graph(-output, 'Self Implementation')

    # Using SKLEARN LDA
    output = do_sklearnLDA(X, y, class_count=2)
    plot_graph(-output, 'SKLEARN Implementation')


if __name__ == '__main__':
    main()