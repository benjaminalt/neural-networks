#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.logistic_layer import LogisticLayer
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)
                                        
    # myLRClassifier = LogisticRegression(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
    hidden_layers = [LogisticLayer(128, 32, isClassifierLayer=True) for layer in range(1)]
    mlp = MultilayerPerceptron(data.trainingSet, data.validationSet, data.testSet, hidden_layers,
                               learningRate=0.005, epochs=30)

    # Train the classifiers
    #print("=========================")
    print("Training...")
    
    # print("\nLogistic Regression has been training..")
    # myLRClassifier.train()
    # print("Done..")

    print("Training MLP...")
    mlp.train()
    print("Done.")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    # perceptronPred = myPerceptronClassifier.evaluate()
    # lrPred = myLRClassifier.evaluate()
    mlpPred = mlp.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    # print("Result of the stupid recognizer:")
    #     # #evaluator.printComparison(data.testSet, stupidPred)
    #     # evaluator.printAccuracy(data.testSet, stupidPred)
    #     #
    #     # print("\nResult of the Perceptron recognizer:")
    #     # #evaluator.printComparison(data.testSet, perceptronPred)
    #     # evaluator.printAccuracy(data.testSet, perceptronPred)
    #     #
    #     # print("\nResult of the Logistic Regression recognizer:")
    #     # #evaluator.printComparison(data.testSet, lrPred)
    #     # evaluator.printAccuracy(data.testSet, lrPred)

    print("Result of the MLP recognizer:")
    evaluator.printComparison(data.testSet, mlpPred)
    evaluator.printAccuracy(data.testSet, mlpPred)

    # Draw
    plot = PerformancePlot("Logistic Regression validation")
    # plot.draw_performance_epoch(myLRClassifier.performances,
    #                             myLRClassifier.epochs)
    plot.draw_performance_epoch(mlp.performances, mlp.epochs)
    
    
if __name__ == '__main__':
    main()
