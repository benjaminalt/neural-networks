import unittest
from model.mlp_neuron import MLPNeuron
from util.activation_functions import Activation

class TestMLPNeuron(unittest.TestCase):

    def test_compute_output(self):
        num_weights = 3
        neuron = MLPNeuron(num_weights, activation="sigmoid", bias=False)
        neuron.weights = [0 for i in range(num_weights)]
        input = [i for i in range(num_weights)]
        activation = Activation.getActivation("sigmoid")
        self.assertEqual(0.5, neuron.compute_output(input))

        neuron.weights = [1 for i in range(num_weights)]
        self.assertAlmostEqual(neuron.compute_output(input), 0.9526, 4)


if __name__ == '__main__':
    unittest.main()
