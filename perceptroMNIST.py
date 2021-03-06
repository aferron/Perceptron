# Perceptron to learn identification of hand-written numbers
# based on MNIST dataset
# CS545

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools as itertools


# Run the perceptron on the test data
def run_test(m_test, weights, x_test, test_predictions, t_test):
    j = 0
    test_correct = 0
    while j < m_test:
        test_dot_products = np.matmul(weights, x_test[j])
        test_predictions[j] = np.argmax(test_dot_products)
        if test_predictions[j] == t_test[j]:
            test_correct += 1
        j += 1
    return test_correct / m_test


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

# number of epochs to run
epochs = 70

# number of nodes (bias node + 28^2)
n = 785

# eta
eta = .001

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
m = len(x_train)
m_test = len(x_test)

# reshape x and x_test to an array
# normalize the data by dividing by 255
# add an element for bias to training and test data
# reshape array x with m rows (60,000) and an unspecified number of columns
# meaning whatever fits, which is 784 to start
x = np.reshape(x, (m, -1)) / 255

# pad adds a value to an array
# pad the 60,000 * 784 matrix called 'x'
# appending 1 to the beginning of every row (making it 60,000 * 785)
x = np.pad(x, ((0, 0), (1, 0)), 'constant', constant_values=(1, 0))

# do the same to x_test as done to x
x_test = np.reshape(x_test, (m_test, -1)) / 255
x_test = np.pad(x_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 0))

# set starting weights randomly between -0.2 and -0.2
# for a 10 * 785 matrix
weights = np.random.uniform(low=-0.05, high=0.05, size=(10, 785))

# activation
# create an array of dot products that holds 60,000 elements, intialized to 0
dot_products = np.zeros(m)
test_dot_products = np.zeros(m_test)

# create an array of 10 zeros as integers
y = np.zeros((10), dtype=int)

# store all predictions for the confusion matrix
predictions = np.zeros(m)
test_predictions = np.zeros(m_test)

# run it M times on the two datapoints
epoch = 0

# set improvement to 1 to start the loop 
improvement = 1

# Continue while the perceptron is continuing to learn and
# epochs is less than some set number (epochs)
while epoch < epochs and improvement > 0.01:
    i = 0
    correct = 0
    # Go through all the training data
    while i < m:
        # The dot product multiplies each pixel value
        # with the weight for its node and sums these values.
        # There are ten sets of weights and ten dot products.
        dot_products = np.matmul(weights, x[i])

        # the max of the dot products is the picked number
        picked = np.argmax(dot_products)
        predictions[i] = picked

        # if the found number is wrong, train
        if not picked == t[i]:

            # only change the weights if we're not on the 0 epoch
            if epoch != 0:

                # set activation values for the 10 results
                y = np.where(dot_products > 0, 1, 0)

                # get an array as it should be to compare with y
                y_target = np.zeros((10), dtype=int)
                y_target[t[i]] = 1

                # the formula to reset the weights
                diff = np.reshape(y - y_target, (1, 10)).T
                xCol = np.reshape(x[i], (1, n))
                new_weights = eta * np.matmul(diff, xCol)
                weights -= new_weights
        else:
            correct += 1
        i += 1

    accuracy[epoch] = correct / m

    # run the perceptron on the test data
    test_accuracy[epoch] = run_test(m_test, weights, x_test, test_predictions, t_test)
    
    # set improvement to see if we continue
    if epoch == 0:
        improvement = accuracy[epoch] 
    else:
        improvement = accuracy[epoch] - accuracy[epoch - 1]
    epoch += 1

# plot the results
plot_accuracy(accuracy, test_accuracy, epoch)
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cm = confusion_matrix(t_test, test_predictions)
plot_confusion_matrix(cm, target_names)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
# if you want you can disp.plot() here
