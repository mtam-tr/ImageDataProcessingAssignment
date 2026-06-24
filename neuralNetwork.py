import numpy as np


class Neuron():
    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        self.weights = np.random.uniform(-1, 1, number_of_inputs)
        if number_of_inputs > 0:
            self.bias = np.random.uniform(-1, 1)


class Layer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.number_of_neurons = number_of_neurons
        self.neurons = np.full(number_of_neurons, None)
        for i in range(number_of_neurons):
            self.neurons[i] = Neuron(number_of_inputs_per_neuron)


class Neural_network():
    # Activation functions from the lesson template, with a local sigmoid so the
    # assignment does not depend on scipy being installed.
    arctan = lambda x: np.arctan(x)
    id_ = lambda x: x
    id_courb = lambda x: 0.5 * ((x**2 + 1)**0.5 - 1) + x
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    sin = lambda x: np.sin(x)
    sinc = lambda x: np.sinc(x)
    softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    softsign = lambda x: x / (1 + np.abs(x))
    swish = lambda x: x * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))
    tanh = lambda x: np.tanh(x)

    activation_functions_dict = {
        'arctan': arctan,
        'id': id_,
        'id_courb': id_courb,
        'sigmoid': sigmoid,
        'sin': sin,
        'sinc': sinc,
        'softmax': softmax,
        'softplus': softplus,
        'softsign': softsign,
        'swish': swish,
        'tanh': tanh,
    }

    def __init__(self, layer_sizes):
        # Example: [2, 8, 3] means 2 inputs, 8 hidden neurons, 3 outputs.
        self.layer_sizes = layer_sizes
        self.number_of_layers = len(layer_sizes)
        self.number_of_inputs = layer_sizes[0]
        self.number_of_outputs = layer_sizes[-1]

        self.layers = np.full(self.number_of_layers, None)
        self.layers[0] = Layer(self.number_of_inputs, 0)
        for i in range(1, self.number_of_layers):
            self.layers[i] = Layer(layer_sizes[i], layer_sizes[i - 1])

    def get_number_of_weights(self):
        number_of_weights = 0
        for i in range(self.number_of_layers - 1):
            number_of_weights += self.layer_sizes[i] * self.layer_sizes[i + 1]
        return number_of_weights

    def get_number_of_biases(self):
        return int(np.sum(self.layer_sizes[1:]))

    def set_weights(self, weights):
        iterator = 0
        for i in range(1, self.number_of_layers):
            for j in range(self.layers[i].number_of_neurons):
                for k in range(self.layers[i].neurons[j].number_of_inputs):
                    self.layers[i].neurons[j].weights[k] = weights[iterator]
                    iterator += 1

    def set_biases(self, biases):
        iterator = 0
        for i in range(1, self.number_of_layers):
            for j in range(self.layers[i].number_of_neurons):
                self.layers[i].neurons[j].bias = biases[iterator]
                iterator += 1

    def get_weights(self):
        weights = [None] * self.get_number_of_weights()
        iterator = 0
        for i in range(1, self.number_of_layers):
            for j in range(self.layers[i].number_of_neurons):
                for k in range(self.layers[i].neurons[j].number_of_inputs):
                    weights[iterator] = self.layers[i].neurons[j].weights[k]
                    iterator += 1
        return weights

    def get_biases(self):
        biases = [None] * self.get_number_of_biases()
        iterator = 0
        for i in range(1, self.number_of_layers):
            for j in range(self.layers[i].number_of_neurons):
                biases[iterator] = self.layers[i].neurons[j].bias
                iterator += 1
        return biases

    def update(self, inputs, activation_functions=None):
        if activation_functions is None:
            activation_functions = [None] + ['id'] * (self.number_of_layers - 1)

        inputs = np.asarray(inputs, dtype=float)
        outputs = inputs

        for i in range(1, self.number_of_layers):
            layer_outputs = []
            for j in range(self.layers[i].number_of_neurons):
                netto_input = 0.0
                for k in range(self.layers[i].neurons[j].number_of_inputs):
                    netto_input += self.layers[i].neurons[j].weights[k] * outputs[k]
                netto_input += self.layers[i].neurons[j].bias
                layer_outputs.append(netto_input)

            activation_function = activation_functions[i]
            if activation_function == 'softmax':
                outputs = self.calc_activation('softmax', np.asarray(layer_outputs))
            else:
                outputs = np.asarray([
                    self.calc_activation(activation_function, value)
                    for value in layer_outputs
                ])

        return outputs.tolist()

    def calc_activation(self, activation_function, x):
        return Neural_network.activation_functions_dict[activation_function](x)
