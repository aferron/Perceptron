from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools as itertools


# number of epochs to run
epochs = 4

# number of nodes (bias node + 28^2)
n = 785

# eta
eta = .001

# number of perceptrons, one per digit
digits = 10

# accurate reads
correct = 0

# array to plot accuracy
accuracy = np.zeros((epochs + 1), dtype=float)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x = np.asarray(x_train)
t = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

# max number of times to run through the training data/test data
m = len(x_train)
m_test = len(x_test)

# add bias, etc to training and test data
x = np.reshape(x, (m, -1)) / 255
x = np.pad(x, ((0,0),(1,0)), 'constant', constant_values=(1, 0))
x_test = np.reshape(x_test, (m_test, -1)) / 255
x_test = np.pad(x_test, ((0,0),(1,0)), 'constant', constant_values=(1, 0))


#set starting weights
weights = np.random.uniform(low=-0.2, high=0.2, size=(10,785))

# activation
dotProducts = np.zeros(m)
y = np.zeros((10), dtype=int)

# store all predictions for the confusion matrix
predictions = np.zeros(m)

# run it M times on the two datapoints
i = 0
epoch = 0

while epoch < epochs:
  while i < m:        
    # get the dot product
    #dotProducts = np.dot(x[i], weights)
    dotProducts = np.matmul(weights, x[i])
    picked = np.argmax(dotProducts)
    predictions[i] = picked

    # if the found number is wrong, train
    if not picked == t[i]:

      # set activation values for the 10 results
      y = np.where(dotProducts > 0, 1, 0)

      # get an array as it should be to compare with y 
      yTarget = np.zeros((10), dtype=int)
      yTarget[t[i]] = 1
      diff = np.reshape(y - yTarget,(1,10)).T
      xCol = np.reshape(x[i],(1,n))
      newWeights = eta * np.matmul(diff,xCol)
      weights = weights - newWeights
    else:
      correct += 1
    i += 1  

  accuracy[epoch] = correct / m
  correct = 0
  i = 0
  epoch += 1

dotProducts = np.zeros(m_test)
predictions = np.zeros(m_test)

# test the accuracy of the perceptron
while i < m_test:
  dotProducts = np.matmul(weights, x_test[i])
  prediction = np.argmax(dotProducts)
  predictions[i] = prediction
  if prediction == y_test[i]:
    correct += 1
  i += 1

accuracy[epoch] = correct / m_test
epoch_range = np.arange(epochs + 1)

print(accuracy)
plt.plot(epoch_range, accuracy, scaley=False)


target_names = ['0','1', '2', '3', '4','5','6','7','8','9']
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
#disp.plot() 


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy    
  if cmap is None:
    cmap = plt.get_cmap('Blues')    
  plt.figure(figsize=(10,10))
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

plot_confusion_matrix(cm,target_names)
