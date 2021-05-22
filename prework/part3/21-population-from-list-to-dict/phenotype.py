import gym
import numpy
import random
from statistics import mean, stdev, pstdev
from itertools import count
from copy import deepcopy
import time
from dataclasses import dataclass, field
from os import path, makedirs
import csv
import json
from dacite import from_dict
from genome import Genome
from config import Config
@dataclass
class Phenotype:
    """Weights are  layer1-layer2 : 2d list of weights for matrix multiplication"""
    """Biases are  layer : 1d list of biases"""
    weights: dict
    biases: dict


def develop_genome_to_neural_network_phenotype(genome: Genome):
    ant_layers = len(genome.layer_chromosome)
    biases = {}
    for receiving_layer in range(1, ant_layers):
        receiving_neruons = [key for (key, neuron) in genome.neuron_chromosome.items() if
                             neuron.parent_layer == receiving_layer]
        # dont need to care about matching weight and bias order yet, since no noeruans are added or removed at the moment
        biases[receiving_layer] = numpy.array([genome.bias_chromosome[key].value for key in receiving_neruons])

    weights = {}
    transmitting_neruons = [key for (key, neuron) in genome.neuron_chromosome.items() if neuron.parent_layer == 0]
    for receiving_layer in range(1, ant_layers):
        receiving_neruons = [key for (key, neuron) in genome.neuron_chromosome.items() if
                             neuron.parent_layer == receiving_layer]
        layer_weights = []
        for receiving_neruon in receiving_neruons:
            layer_weights.append([weight_gene.value for weightKey, weight_gene in
                                  genome.weight_chromosome.items() if (
                                          weight_gene.receiving_neuron == receiving_neruon and
                                          weight_gene.transmitting_neuron in transmitting_neruons)])
        weights["{}-{}".format(receiving_layer - 1, receiving_layer)] = numpy.array(layer_weights)
        transmitting_neruons = receiving_neruons

        # filter(lambda keyValue : keyValue[1].transmitting_neuron == receiving_layer-1 and keyValue[1].receiving_neuron == receiving_layer ,genome.weight_chromosome.items()))

    return Phenotype(weights=weights, biases=biases)



class NeuralNetworkModel:
    def __init__(self, phenotype: Phenotype,config: Config,id=None):

        self.weights = phenotype.weights

        # self.biases = numpy.array(genome["biases"])
        self.biases = phenotype.biases
        self.layer_sizes = config.layer_sizes
        self.config = config

    def noise_output(self, ant_output):
        # return random.choice(range(3))
        return numpy.random.uniform(-1, 1, size=ant_output)

    def run(self, input_values):
        if self.config.use_action_noise:
            if random.random() < self.config.action_noise_rate:
                return self.noise_output(ant_output=self.layer_sizes[-1])

        layer_values = [input_values]
        for layer in range(1, len(self.layer_sizes)):
            incoming_values = layer_values[-1]
            neruon_sums = numpy.matmul(self.weights["{}-{}".format(layer - 1, layer)], incoming_values) + self.biases[
                layer]
            neruon_sums = numpy.tanh(neruon_sums)
            layer_values.append(neruon_sums)

        # action_choise = numpy.argmax(layer_values[-1])
        action_choise = layer_values[-1]
        # action_choise = numpy.max(layer_sizes[-1], (1))
        action_choise = numpy.tanh(action_choise)

        return action_choise

        # https://www.youtube.com/watch?v=lFOOjeH2wsY
        # https://www.youtube.com/watch?v=woa34ugDSwY
        # https://www.youtube.com/watch?v=wasZ0MusbdM&list=PL_mqLx7AmDzeG5kXYbhllIaLiZIALla3P&index=5
        # todo clamp / normalize outputs  with activation function

