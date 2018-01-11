# Implementation of Artificial Neural Network

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# Load the Iris Dataset
def load_data():
    return datasets.load_iris()


# Process the Dataset
def process_data(main_input_data):

    # Perform scaling on the features

    temp_x = main_input_data.data[50:,2:]

    temp_x[:, 0] = (temp_x[:, 0] - np.min(temp_x[:, 0])) / (np.max(temp_x[:, 0]) - np.min(temp_x[:, 0]))
    temp_x[:, 1] = (temp_x[:, 1] - np.min(temp_x[:, 1])) / (np.max(temp_x[:, 1]) - np.min(temp_x[:, 1]))

    X = temp_x

    # Recaliberate Y

    y = main_input_data.target[50:]

    for i in range(len(main_input_data.target[50:])):
        if i < 50:
            y[i] = 0
        else:
            y[i] = 1

    y = y.reshape(100, 1)

    return X, y


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid Function
def ddx_sigmoid(x):
    return x * (1 - x)


# NEURAL NETWORK
def neural_network(X, y, iterations_max):

    # Learning Rate
    alpha = 0.01

    # Number of Neruons in Input Layer = Number of Features in the data set
    inputlayer_neurons = X.shape[1]

    # Number of Neurons in the Hidden Layer
    hiddenlayer_neurons = 2

    # Number of Neurons at the Output Layer
    output_neurons = 1

    # Weight and Bias Initialization
    theta_to_hidden = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    bias_to_hidden = np.random.uniform(size=(1, hiddenlayer_neurons))
    theta_to_output = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    bias_to_output = np.random.uniform(size=(1, output_neurons))

    #####   TRAINING - BEGIN  #####
    cost_fn = []
    print "\n\tIterations:\t",
    for i in range(iterations_max):

        #####   Forward Propagation - BEGIN   #####

        # Input to Hidden Layer = (Dot Product of Input Layer and Weights) + Bias
        hidden_layer_input = (np.dot(X, theta_to_hidden)) + bias_to_hidden

        # Activation of input to Hidden Layer by using Sigmoid Function
        hiddenlayer_activation = sigmoid(hidden_layer_input)

        # Input to Output Layer = (Dot Product of Hidden Layer Activations and Weights) + Bias
        output_layer_input = np.dot(hiddenlayer_activation, theta_to_output) + bias_to_output

        # Activation of input to Output Layer by using Sigmoid Function
        output = sigmoid(output_layer_input)

        #####   Forward Propagation - END #####

        #####   Backward Propagation - BEGIN   #####

        # Error at output layer
        error_at_output = y - output

        # FIND THE COST PER ITERATION
        if (i % (iterations_max / 10) == 0):
            print i,
            cost_fn.append((-1 / len(X)) * np.sum((y * np.log(output) + (1 - y) * np.log(1 - output))))

        # Finding the 'Gradient Descent' at each step for Output Layer and Hidden Layer
        slope_output_layer = ddx_sigmoid(output)
        slope_hidden_layer = ddx_sigmoid(hiddenlayer_activation)

        # Delta at Output Layer
        delta_at_output = error_at_output * slope_output_layer

        # Error at hidden layer
        Error_at_hidden_layer = delta_at_output.dot(theta_to_output.T)

        # Delta at hidden layer
        delta_at_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

        # Update the weights to output
        theta_to_output += hiddenlayer_activation.T.dot(delta_at_output) * alpha
        bias_to_output += np.sum(delta_at_output, axis=0, keepdims=True) * alpha

        # Update the weights to input
        theta_to_hidden += X.T.dot(delta_at_hiddenlayer) * alpha
        bias_to_hidden += np.sum(delta_at_hiddenlayer, axis=0, keepdims=True) * alpha

        #####   Backward Propagation - END   #####
    #####   TRAINING - END  #####
    print "\n"

    model = {'theta_to_hidden': theta_to_hidden,
             'bias_to_hidden': bias_to_hidden,
             'theta_to_output': theta_to_output,
             'bias_to_output': bias_to_output,
             'output': output}

    return model


# PREDICT FUNCTION
def predict(input, trained_model):
    pred_hidden_layer_input = (np.dot(input, trained_model['theta_to_hidden'])) + trained_model['bias_to_hidden']
    pred_hiddenlayer_activations = sigmoid(pred_hidden_layer_input)
    pred_output_layer_input = np.dot(pred_hiddenlayer_activations, trained_model['theta_to_output']) + \
                              trained_model['bias_to_output']
    pred_output = sigmoid(pred_output_layer_input)

    # Setting the Threshold as 0.5
    if pred_output >= 0.5:
        pred_output = 1
    else:
        pred_output = 0

    return pred_output

# LEAVE ONE OUT ANALYSIS
def leave_one_out(X, y, iterations):
    error = 0
    correct = []
    wrong = []
    for i in range(100):

        # Make a copy of the dataset
        temp_x = X
        temp_y = y

        # Create Test Data
        test_input = temp_x[i]
        test_output = temp_y[i]

        # Delete test data from actual dataset
        temp_x = np.delete(temp_x, i, axis=0)
        temp_y = np.delete(temp_y, i, axis=0)

        # Running the Neural Network on the training dataset
        print "\nSample ", i + 1, " "
        model = neural_network(temp_x, temp_y, iterations_max=iterations)

        # Predicting the output on the test data
        predicted_output = predict(test_input, model)

        # Increase Error Count
        if (predicted_output != test_output):
            error += 1
            wrong.append(test_input)
        else:
            correct.append(test_input)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Artificial Neural Network Predictions')
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.scatter(np.array(wrong)[:,0], np.array(wrong)[:,1], color='red', label='Incorrect')
    ax.scatter(np.array(correct)[:, 0], np.array(correct)[:, 1], color='blue', label='Correct')
    ax.legend()
    plt.savefig('plots/NeuralNetwork', dpi=200)

    plt.close()

    return error


# MAIN FUNCTION
def main():

    # Load Data
    main_input_data = load_data()

    # Process the Data to get final input and output data to feed to Neural Network
    X, y = process_data(main_input_data)

    # Run a 'Leave-one-out analysis' for 100 samples to find the error rate
    error = float(leave_one_out(X, y, iterations=10001))

    print "\nError for 100 samples is:\t", int(error)
    error_rate = error / 100
    print "\nAverage Error Rate is:\t", error_rate


if __name__ == '__main__':
    main()

'''
##################  OUTPUT  ##################
Sample  1  
	Iterations:	0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 
...
Sample  100  
	Iterations:	0 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 
Error for 100 samples is:	7
Average Error Rate is:	0.07
'''