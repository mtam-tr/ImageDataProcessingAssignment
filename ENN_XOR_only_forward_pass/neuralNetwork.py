import numpy as np
from scipy.special import expit

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
    def __init__(self, layer_sizes):   
        self.layer_sizes = layer_sizes
        self.number_of_layers = len(layer_sizes)
        self.number_of_inputs = layer_sizes[0]
        self.number_of_outputs = layer_sizes[-1] 
        self.layers = np.full(self.number_of_layers, None)     
        self.layers[0] = Layer(self.number_of_inputs, 0)
        
        for i in range(1, self.number_of_layers):
            self.layers[i] = Layer(layer_sizes[i], layer_sizes[i - 1])
	
    def set_weights(self, weights):
        iterator = 0
        for i in range(1, self.number_of_layers):
            for j in range(0, self.layers[i].number_of_neurons):
                for k in range(0, self.layers[i].neurons[j].number_of_inputs):
                    self.layers[i].neurons[j].weights[k] = weights[iterator]
                    iterator += 1         
	
    def set_biases(self, biases):
        iterator = 0
        for i in range(1, self.number_of_layers):
            for j in range(0, self.layers[i].number_of_neurons):
                self.layers[i].neurons[j].bias = biases[iterator]
                iterator += 1
    
    def update(self, inputs):
        
        outputs = [None]
        
        for i in range(1, self.number_of_layers):
            if(i > 1):
                inputs = outputs
			
            number_of_neurons_in_layer = self.layers[i].number_of_neurons            
            outputs = [None] * number_of_neurons_in_layer
            
            for j in range(number_of_neurons_in_layer):
                netto_input = 0
				
                number_of_inputs = self.layers[i].neurons[j].number_of_inputs
                for k in range(number_of_inputs):
                    netto_input += self.layers[i].neurons[j].weights[k] * inputs[k]
				
                # Add bias
                netto_input += self.layers[i].neurons[j].bias
				
                # Calculate output
                outputs[j] = self.calc_activation(netto_input)
                
        return outputs
	
    def calc_activation(self, x):
        return expit(x)