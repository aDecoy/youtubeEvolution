import random
import numpy
from genome import Genome
from dataclasses import dataclass
from config import Config

@dataclass
class Phenotype:
    weights: dict
    biases: dict


def develop_genome_to_phenotype(genome: Genome):
    ant_layers = len(genome.layer_chromosome)
    biases = {}
    for receiving_layer in range(1, ant_layers):
        receiving_neurons = [key for (key, neuron) in genome.neuron_chromosme.items() if neuron.parent_layer == receiving_layer]
        biases[receiving_layer] = numpy.array([genome.bias_chromosme[key].value for key in receiving_neurons])

    weights = {}
    transmitting_neurons = [key for (key, neuron) in genome.neuron_chromosme.items() if neuron.parent_layer == 0]
    for receiving_layer in range(1, ant_layers):
        receiving_neurons = [key for (key, neuron) in genome.neuron_chromosme.items() if neuron.parent_layer == receiving_layer]
        layer_weights = []
        for receiving_neuron in receiving_neurons:
            layer_weights.append([weight_gene.value for weightKey, weight_gene in genome.weight_chromosme.items()
                                  if (weight_gene.receiving_neuron == receiving_neuron and
                                      weight_gene.transmitting_neuron in transmitting_neurons)])

        weights["{}-{}".format(receiving_layer - 1, receiving_layer)] = numpy.array(layer_weights)
        transmitting_neurons = receiving_neurons

    return Phenotype(weights=weights, biases=biases)


class NeuralNetworkModel:

    # def __init__(self, genome={"weights": [[2, 0.9, 1.2, 1.1, 1, 1, 1, 1]], "biases": [0, 0, 0, 0, 0, 0, 0, 0, ]}):
    def __init__(self, phenotype: Phenotype, config : Config):
        # 8 input, 4 hidden , 1 output
        # if genome is None:

        self.weights = phenotype.weights
        self.biases = phenotype.biases
        self.config = config

    def noise_output(self):
        return random.choice(range(3))

    def run(self, input_values):

        if self.config.use_action_noise:
            if random.random() < self.config.action_noise_rate:
                return self.noise_output()

        input_values = numpy.array(input_values).reshape(-1)
        layer_outputs = [input_values]
        for i in range(1,len(self.config.layer_sizes)):
            layer_outputs.append( numpy.matmul(self.weights["{}-{}".format(i-1,i)], layer_outputs[i-1]) + self.biases[i])
        # best_action = numpy.argmax(layer_outputs[-1])
        best_action = layer_outputs[-1]
        return best_action


