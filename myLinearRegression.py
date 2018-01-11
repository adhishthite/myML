# Implement Linear Regression

import numpy as np
from numpy import genfromtxt as GFT
import matplotlib.pyplot as plt


# MAIN FUNCTION
def main():
    # Load Data
    input_data = load_data()

    # Perform Linear Regression
    lin_reg_result = do_linear_reg(input_data)

    # Plot Data
    plot_data(lin_reg_result)

    exit()


# LOAD THE INPUT DATA
def load_data():
    # Original Dataset
    input_data = GFT('datasets/linear_regression_test_data.csv', skip_header=1, delimiter=',')

    # Delete first column
    input_data = np.delete(input_data, 0, axis=1)

    return input_data


# IMPLEMENT LINEAR REGRESSION ON INPUT
def do_linear_reg(input_data):
    x = input_data[:, 0]
    y = input_data[:, 1]

    # Length of input data
    n = len(x)

    # Calculate 'Mean' of Input
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    # Estimation of beta0 and beta1

    s_yx = np.sum((y - y_bar) * (x - x_bar))
    s_xx = np.sum((x - x_bar) ** 2)

    beta1_hat = s_yx / s_xx
    beta0_hat = y_bar - (beta1_hat * x_bar)

    # Calculate y_hat value
    y_hat = beta0_hat + beta1_hat * x

    lin_reg_result = {'input_data': input_data,
                      'x': x,
                      'y_hat': y_hat}

    return lin_reg_result


# PLOT DATA ON GRAPH
def plot_data(lin_reg_result):
    # Plotting the Original Data

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    plt.legend(loc='upper left')
    plt.suptitle('Linear Regression')

    # y vs x
    ax.scatter(lin_reg_result['input_data'][:, 0], lin_reg_result['input_data'][:, 1], color='blue',
               label='Input Data : \'y\' vs \'x\'')

    # y_theoretical vs x
    ax.scatter(lin_reg_result['input_data'][:, 0], lin_reg_result['input_data'][:, 2], color='red',
               label='Input Data : \'y_theoretical\' vs \'x\'')

    # Linear Regression
    ax.plot(lin_reg_result['x'], lin_reg_result['y_hat'], color='orange', linewidth=1,
            label='Linear Regression Line')

    plt.legend(loc='upper left')
    plt.savefig('plots/LinearRegression.jpg')
    plt.close()


if __name__ == '__main__':
    main()
