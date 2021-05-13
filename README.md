# Perceptron
Trains a perceptron to recognize handwritten numbers using the MNIST data set


The MNIST data set contains 60,000 training images with handwritten numbers stored in 28 x 28 pixel images. Each pixel contains a number indicating its shading ranging from white to black. To train with this data the 784 pixels are taken as inputs, plus one for the bias to give 10 outputs indicating the number the perceptron calculates that is written. The accuracy of this output is checked, and weights for each input are adjusted to improve performance on the next epoch. As the process continues, the perceptron "learns" to identify the numbers with increasing accuracy.

For this project, the training and test data was processed with with the learning rate set to 0.001, 0.025, 0.3, and 0.9. Four epochs were used to train the data for each learning rate. Both input data sets were normalized by dividing by 255, then an additional row of 1's was added to each matrix for the bias. For each image, the max of the outputs was selected as the digit. If this didn't match the correct answer the weights were updated before processing the next image. The accuracy for each epoch was recorded and plotted along with the results for the test data. A confusion matrix was produced for the test results for each learning rate. Graphs of accuracy for each epoch and confusion matrices can be seen in Perceptron.pdf.