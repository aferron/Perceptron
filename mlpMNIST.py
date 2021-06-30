# Perceptron to learn identification of hand-written numbers
# based on MNIST dataset
# CS545

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools as itertools


# Run the perceptron on the test data
def run_test(num_test_examples, weights, x_test, test_predictions, t_test):
    j = 0
    test_correct = 0
    while j < num_test_examples:
        test_dot_products = np.matmul(weights, x_test[j])
        test_predictions[j] = np.argmax(test_dot_products)
        if test_predictions[j] == t_test[j]:
            test_correct += 1
        j += 1
    return test_correct / num_test_examples


# Plot the accuracy of the training and test results
def plot_accuracy(accuracy, test_accuracy, epochs):
    # remove extra entries from accuracy arrays
    accuracy = np.trim_zeros(accuracy)
    print("accuracy: ", accuracy)
    if len(accuracy) != epochs:
        sys.exit("accuracy array doesn't match length of epochs")

    test_accuracy = np.trim_zeros(test_accuracy)
    print("test_accuracy: ", test_accuracy)
    if len(test_accuracy) != epochs:
        sys.exit("test_accuracy array doesn't match length of epochs")

    epoch_range = np.arange(epochs)

    print(accuracy)
    plt.plot(epoch_range, accuracy, label='train', scaley=False)
    plt.plot(epoch_range, test_accuracy, label='test', scaley=False)
    plt.legend(loc='lower right')


# Plot the confusion matrix
# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# number of hidden units
n = 20

# number of epochs to run
epochs = 5

# batch size
batch_size = 32

# momentum
alpha = 0.9

# number of nodes (bias node + 28^2)
num_input_nodes = 785

# learning rate 
eta = .1

# number of perceptrons, one per digit
digits = 10

# array to plot accuracy
accuracy = np.zeros((epochs + 1), dtype=float)
test_accuracy = np.zeros((epochs + 1), dtype=float)


# load MNIST data set
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# save the training and testing data sets in to arrays
# x and t are for training, x_test and y_test are for testing
x = np.asarray(x_train)
t = np.asarray(y_train)
x_test = np.asarray(x_test)
t_test = np.asarray(y_test)

# max number of times to run through the training data/test data
num_train_examples = len(x_train)
num_test_examples = len(x_test)

# reshape x and x_test to an array
# normalize the data by dividing by 255
# add an element for bias to training and test data
# reshape array x with num_train_examples rows (60,000) and an unspecified number of columns
# meaning whatever fits, which is 784 to start
x = np.reshape(x, (num_train_examples, -1)) / 255

# pad the 60,000 * 784 matrix called 'x'
# appending 1 to the beginning of every row (making it 60,000 * 785)
x = np.pad(x, ((0, 0), (1, 0)), 'constant', constant_values=(1, 0))

# do the same to x_test as done to x
x_test = np.reshape(x_test, (num_test_examples, -1)) / 255
x_test = np.pad(x_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 0))

# set starting weights randomly between -0.2 and -0.2
# for a 10 * 785 matrix
weights = np.random.uniform(low=-0.05, high=0.05, size=(10, 785))

# store all predictions for the confusion matrix
test_predictions = np.zeros(num_test_examples)

# store error for each training example in a matrix holding
# all error for the batch
output_error = np.zeros(digits)

# run it M times on the two datapoints
epoch = 0


# Continue while the perceptron is continuing to learn and
# epochs is less than some set number (epochs)
while epoch < epochs:
    i = 0
    correct = 0
    # Go through all the training data
    while i < num_train_examples:
        j = 0
        # The dot product multiplies each pixel value
        # with the weight for its node and sums these values.
        # There are ten sets of weights and ten dot products.
        # weights is [10, 785]
        # x[i] is [1, 785]
        # dot_products is [10, 1]
        # o_activation is [10, 1]
        dot_products = np.matmul(weights, x[i])
        o_activation = 1/(1 + np.exp(-dot_products)) 

        # the max of the activations is the picked number
        picked = np.argmax(o_activation)

        # if the found number is wrong, train
        if not picked == t[i]:

            # only change the weights if we're not on the 0 epoch
            if epoch != 0:
                # get an array as it should be to compare with o_activation
                # y_target is [1, 10]
                y_target = np.full(10, 0.1)
                y_target[t[i]] = 0.9

                # compute and store error
                # output_error is [1, 10]
                output_error = o_activation * np.matmul(1 - o_activation, y_target - o_activation)

                # the formula to reset the weights
                # diff is [10, 785]
                # output_error = np.reshape(output_error, (1, 10)).T
                weights += eta * np.matmul(np.reshape(output_error, (10, 1)), np.reshape(x[i], (1, num_input_nodes))) # + alpha * prev_diff

                # diff = np.reshape(o_activation - y_target, (1, 10)).T
                # xCol = np.reshape(x[i], (1, num_input_nodes))
                # weights -= eta * np.matmul(diff, xCol)
        else:
            correct += 1
        i += 1

    accuracy[epoch] = correct / num_train_examples

    # run the perceptron on the test data
    test_accuracy[epoch] = run_test(num_test_examples, weights, x_test, test_predictions, t_test)
    
    epoch += 1

# plot the results
plot_accuracy(accuracy, test_accuracy, epoch)
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cm = confusion_matrix(t_test, test_predictions)
plot_confusion_matrix(cm, target_names)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
# if you want you can disp.plot() here
