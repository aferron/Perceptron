# Perceptron to learn identification of hand-written numbers
# based on MNIST dataset
# CS545

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools as itertools



# number of epochs to run
epochs = 100

# batch size
batch_size = 2

# momentum
alpha = 0.9

# number of nodes (bias node + 28^2)
num_input_nodes = 3

# number of hidden nodes
num_hidden_nodes = 2

# learning rate 
eta = 0.1

# number of perceptrons, one per digit
digits = 2

# array to plot accuracy
accuracy = np.zeros((epochs + 1), dtype=float)
test_accuracy = np.zeros((epochs + 1), dtype=float)


# save the training and testing data sets in to arrays
# x and t are for training, x_test and y_test are for testing
x = [[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 0, 1]]
t = [1, 1, 0, 1]

# max number of times to run through the training data/test data
num_train_examples = 4
num_test_examples = 0

# set starting input_weights randomly 
# input_weights is [num_hidden_nodes, 785]
# hidden_weights is [10, num_hidden_nodes]
# # input_weights = np.random.uniform(low=-0.05, high=0.05, size=(num_hidden_nodes, num_input_nodes))
# hidden_weights = np.random.uniform(low=-0.05, high=0.05, size=(10, num_hidden_nodes))
input_weights = np.array([[0.5, 0.1, -0.5], [-0.2, 0.3, 0.4]])
hidden_weights = np.array([[-0.2, 0.1], [-0.1, -0.5]])

# store all predictions for the confusion matrix
test_predictions = np.zeros(num_test_examples)

epoch = 0

# Continue while 
# epochs is less than some set number
while epoch < epochs:
    i = 0
    correct = 0
    # Go through all the training data
    while i < num_train_examples:
        # store error for each training example in a matrix holding
        # all error for the batch
        output_error = np.zeros(digits)
        hidden_error = np.zeros(num_hidden_nodes)
        # The dot product multiplies each pixel value with the weight for its node and sums these values.
        # hidden_dot_products is [num_hidden_nodes, 1]
        # hidden_activation is [num_hidden_nodes, 1]
        hidden_dot_products = np.matmul(x[i], input_weights.transpose())
        print(hidden_dot_products)
        hidden_activation = 1/(1 + np.exp(-hidden_dot_products))
        print(hidden_activation)

        # do the same on the hidden layer
        # output_dot_products is [10, 1]
        # output_activation is [10, 1]
        output_dot_products = np.matmul(hidden_activation, hidden_weights.transpose())
        print(output_dot_products)
        output_activation = 1/(1 + np.exp(-output_dot_products)) 
        print(output_activation)

        # the max of the activations is the picked number
        picked = np.argmax(output_activation)
        print(picked)
        print(t[i])


        # print("\n\ni: ", i, "\noutput dot products: ", output_dot_products,
                        # "\nmax of dot products: ", np.argmax(output_dot_products), "   picked: ", picked, "target: ", t[i])

        if picked == t[i]:
            correct += 1

        # get an array as it should be to compare with output_activation
        # y_target is [1, 10]
        y_target = np.full(2, 0.1)
        y_target[t[i]] = 0.9
        print(y_target)

        # compute and store error
        # output_error is [1, 2]
        # hidden_error is [1, num_hidden_nodes]
        diff1 = y_target - output_activation
        print(diff1)
        diff2 = 1 - output_activation
        print(diff2)
        print(np.matmul(diff2, diff1))
        print(np.multiply(output_activation, np.multiply(1 - output_activation, y_target - output_activation)))
        output_error = np.multiply(output_activation, np.multiply(1 - output_activation, y_target - output_activation))
        print(output_error)


        sum = np.matmul(output_error, hidden_weights)
        hidden_error = np.multiply(hidden_activation, np.multiply(1 - hidden_activation, sum))
        print(sum)
        print(hidden_activation)
        print(np.multiply(1 - hidden_activation, sum))



        # update the weights
        # diff is [10, 785]
        # output_error = np.reshape(output_error, (1, 10)).T
        # print("\n\ni: ", i, "   \nbefore updating: \nhidden weights: ", hidden_weights, "\ninput weights: ", input_weights)
        h_to_o_delta_w = eta * np.matmul(np.reshape(output_error, (digits, 1)), np.reshape(hidden_activation, (1, num_hidden_nodes)))
        hidden_weights -= h_to_o_delta_w
        i_to_h_delta_w = eta * np.matmul(np.reshape(hidden_error, (num_hidden_nodes, 1)), np.reshape(x[i], (1, num_input_nodes)))
        input_weights -= i_to_h_delta_w
        # print("\n\nh_to_o_delta_w: ", h_to_o_delta_w)
        # print("i_to_h_delta_w: ", i_to_h_delta_w)
        quit()

        # print("after updating: \nhidden_weights: ", hidden_weights, "input weights", input_weights, "\n")

        i += 1
    accuracy[epoch] = correct / num_train_examples

    # run the perceptron on the test data
    # test_accuracy[epoch] = run_test(num_test_examples,einput_weights, x_test, test_predictions, t_test) 
    epoch += 1

print(accuracy)


# target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# cm = confusion_matrix(t_test, test_predictions)
# plot_confusion_matrix(cm, target_names)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
# if you want you can disp.plot() here
