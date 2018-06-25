
import numpy as np
import argparse

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from data.mnist_seven import MNISTSeven

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : DataSet
        valid : DataSet
        test : DataSet
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : DataSet
        validationSet : DataSet
        testSet : DataSet
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        #self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128,
                                         None, inputActivation, False))

        self.layers.extend(layers)

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(32, 10,
                                         None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        # Input layer
        next_inp = inp
        for layer in self.layers:
            next_inp = layer.forward(next_inp)
            next_inp = np.insert(next_inp, 0, 1) # Add bias

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the errors
        """
        return self.loss.calculateError(target, self._get_output_layer().outp)
    
    def _update_weights(self, learning_rate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learning_rate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            self._log("Epoch {}".format(epoch), verbose)

            for i in range(self.trainingSet.input[0].shape[0]):
                training_example = self.trainingSet.input[i]
                label_idx = self.trainingSet.label[i]
                self._feed_forward(training_example)
                label = np.zeros((10,)) # We one-hot encode output
                label[label_idx] = 1
                output = self._get_output_layer().outp
                delta = self.loss.calculateDerivative(label, output)
                #w = np.ones(shape=(10,))
                w = 1.0
                for hidden_layer in reversed(self.layers):
                    delta = hidden_layer.computeDerivative(delta, w)
                    w = hidden_layer.weights.T
                    # Remove first weight (bias) to fix shape problems
                    w = np.delete(w, 0, 1)
                self._update_weights(self.learningRate)

            # Plotting from logistic_regression
            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)
        return np.argmax(self._get_output_layer().outp)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def _log(self, msg, verbose=True):
        if verbose:
            print msg

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)


def main(args):
    hidden_layers = [LogisticLayer(128, 128, isClassifierLayer=True) for layer in range(args.num_layers)]
    data = MNISTSeven(args.dataset, 3000, 1000, 1000, oneHot=True)
    MLP = MultilayerPerceptron(data.trainingSet, data.validationSet, data.testSet, hidden_layers)
    MLP.train(verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the multilayer perception")
    parser.add_argument("num_layers", type=int, help="Number of hidden layers")
    parser.add_argument("dataset", type=str, help="Path to dataset (CSV file)")
    main(parser.parse_args())