from dataclasses import dataclass
import random
from config import Config


@dataclass
class Genome:
    layer_chromosome: dict = None
    neuron_chromosme: dict = None
    weight_chromosme: dict = None
    bias_chromosme: dict = None


@dataclass
class BiasGene:
    innovationNumber: int
    value: float
    parent_neuron: int

    def mutate(self, config):
        self.value += random.uniform(-config.mutation_power, +config.mutation_power)


@dataclass
class WeightGene:
    innovationNumber: int
    value: float
    transmitting_neuron: int
    receiving_neuron: int

    def mutate(self, config):
        self.value += random.uniform(-config.mutation_power, +config.mutation_power)


@dataclass
class NeruonGene:
    innovationNumber: int
    parent_layer: int

    def mutate(self, config):
        pass


@dataclass
class LayerGene:
    innovationNumber: int

    # later add activation function
    def mutate(self, config):
        pass


def create_new_layer_chromosome(ant_layers):
    return {i: LayerGene(innovationNumber=i) for i in range(ant_layers)}


def create_new_neruon_chromosome(config: Config):
    neurons = {}
    input_innovation_number = -1

    # Input neurons
    for i in range(config.layer_sizes[0]):
        neurons[input_innovation_number] = NeruonGene(innovationNumber=input_innovation_number, parent_layer=0)
        input_innovation_number -= 1

    # Hidden neurons
    for layer, ant_neurons in enumerate(config.layer_sizes[1:-1], start=1):
        for i in range(ant_neurons):
            id = next(config.global_neuron_chromsome_id_counter)
            neurons[id] = NeruonGene(innovationNumber=id, parent_layer=layer)
    # Output neruon
    output_innovation_number = 0
    for i in range(config.layer_sizes[-1]):
        id = output_innovation_number
        output_innovation_number += 1
        neurons[id] = NeruonGene(innovationNumber=id, parent_layer=len(config.layer_sizes) - 1)

    return neurons


def create_new_weight_chromosome(neuron_choromosome: dict, config: Config):
    weights = {}
    # input neurons
    transmitting_neurons = {key: gene for key, gene in neuron_choromosome.items() if gene.parent_layer == 0}

    # Loop over the rest of the layers, finding values for all weights
    for receiving_layer in range(1, len(config.layer_sizes)):
        receiving_neurons = {key: gene for key, gene in neuron_choromosome.items() if gene.parent_layer == receiving_layer}

        # for each gene
        for transmitting_neuron in transmitting_neurons:
            for receiving_neuron in receiving_neurons:
                id = next(config.global_weight_chromsome_id_counter)
                weights[id] = WeightGene(innovationNumber=id, transmitting_neuron=transmitting_neuron, receiving_neuron=receiving_neuron,
                                         value=random.uniform(config.genome_value_min, config.genome_value_max))

        transmitting_neurons = receiving_neurons

    return weights


def create_new_bias_chromosome(neuron_chromosome, config: Config):
    biases = {}
    for neuron_id in neuron_chromosome:
        if neuron_id >= 0:
            biases[neuron_id] = BiasGene(innovationNumber=neuron_id, parent_neuron=neuron_id,
                                         value=random.uniform(config.genome_value_min, config.genome_value_max))
    return biases


def create_new_genome(config: Config):
    layers = create_new_layer_chromosome(len(config.layer_sizes))
    neurons = create_new_neruon_chromosome(config)
    weights = create_new_weight_chromosome(neurons, config=config)
    biases = create_new_bias_chromosome(neuron_chromosome=neurons, config=config)

    return Genome(layer_chromosome=layers,
                  neuron_chromosme=neurons,
                  weight_chromosme=weights,
                  bias_chromosme=biases)
