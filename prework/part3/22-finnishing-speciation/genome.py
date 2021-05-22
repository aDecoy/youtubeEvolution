import random
from dataclasses import dataclass

from config import Config


# def __iter__(self):
#     for attr, value in self.__dict__.iteritems():
#         yield attr, value


# @dataclass
# class Chromosome:
#     """Class for keeping track of an chromosome with name and genes"""
#     locationId: str = None
#     # Key: location to use the genes , value : genes
#     genes: dict = None


@dataclass
class Gene:
    """Class for keeping track of an gene"""
    # value: "AnyTypeWorkaround"
    innovationNumber: str = None


@dataclass
class LayerGene:
    """Class for keeping track of an gene"""
    level: int = None
    innovationNumber: str = None
    # activation_function: str = None
    activation_function: str = None
    can_average_values_in_crossover = False

    def mutate(self, config):
        pass
    def gene_distance(self, other,config :Config):
        if (self.activation_function != other.activation_function):
            return 0 # this is always the same for now
        else:
            return 0


@dataclass
class NeuronGene:
    """Class for keeping track of an gene"""
    innovationNumber: int = None
    parent_layer: str = None
    can_average_values_in_crossover = False

    def mutate(self,config):
        pass

    def gene_distance(self, other,config :Config):
        return 0


@dataclass
class BiasGene:
    """Class for keeping track of an gene"""
    value: float
    parent_neuron: int = None
    innovationNumber: int = None
    can_average_values_in_crossover = True

    def mutate(self,config : Config):
        floatValueMutation(self, config=config)

    def gene_distance(self, other,config : Config):
        return abs(self.value - other.value)*config.compatibility_bias_coefficient


@dataclass
class WeightGene:
    """Class for keeping track of an gene"""
    value: float
    innovationNumber: int = None
    transmitting_neuron: str = None
    receiving_neuron: str = None
    can_average_values_in_crossover = True

    def mutate(self,config : Config):
        floatValueMutation(self,config=config)

    def gene_distance(self, other,config :Config):
        return abs(self.value - other.value)* config.compatibility_weight_coefficient


@dataclass
class Phenotype:
    """Weights are  layer1-layer2 : 2d list of weights for matrix multiplication"""
    """Biases are  layer : 1d list of biases"""
    weights: dict
    biases: dict


@dataclass
class Genome:
    """Class for track of an genome with an id from the individual and a dict of chromosomes"""
    # id: "AnyTypeWorkaround" = None # todo trenger ikke denne i genome.
    # chromosomes: Chromosomes = None
    layer_chromosome: dict = None
    neuron_chromosome: dict = None
    weight_chromosome: dict = None
    bias_chromosome: dict = None

    def get(self, chromosome_key):
        return getattr(self, chromosome_key)



def floatValueMutation(gene: Gene,config: Config):
    if random.random() < config.mutation_rate:
        # Mutate
        gene.value += random.uniform(-config.mutation_power, +config.mutation_power)



def create_new_genome(id, config:Config):
    layer_sizes = config.layer_sizes
    layers = create_new_layer_chromosome(ant_layers=len(layer_sizes))
    neurons = create_new_neruon_chromosome(layer_sizes=layer_sizes,config=config)
    biases = create_new_bias_chromosome(neurons,config=config)
    weights = create_new_weight_chromosome(neuron_chromosome=neurons, ant_layers=len(layer_sizes),config=config)
    genome = Genome(
        layer_chromosome=layers,
        neuron_chromosome=neurons,
        bias_chromosome=biases,
        weight_chromosome=weights)
    return genome


def create_new_layer_chromosome(ant_layers):
    return {i: LayerGene(level=i) for i in range(ant_layers)}


def create_new_neruon_chromosome(layer_sizes, config: Config):
    neruons = {}
    # Input neruons
    input_innovation_number = -1
    for i in range(layer_sizes[0]):
        id = input_innovation_number
        input_innovation_number -= 1
        neruons[id] = NeuronGene(innovationNumber=id, parent_layer=0)
    # Hidden neurons
    for layer, ant_neruons in enumerate(layer_sizes[1:-1], start=1):
        for i in range(ant_neruons):
            id = next(config.global_neruon_id_counter)
            neruons[id] = NeuronGene(innovationNumber=id, parent_layer=layer)
    # Output Neruons
    output_innovation_number = 0
    for i in range(layer_sizes[-1]):
        id = output_innovation_number
        output_innovation_number += 1
        neruons[id] = NeuronGene(innovationNumber=id, parent_layer=len(layer_sizes) - 1)

    return neruons


def create_new_bias_chromosome(neuron_chromosome, config: Config):
    biases = {}
    for neuron_id in neuron_chromosome:
        if neuron_id >= 0:
            biases[neuron_id] = BiasGene(parent_neuron=neuron_id,
                                         value=random.uniform(config.genome_value_min, config.genome_value_max))
    return biases


def create_new_weight_chromosome(neuron_chromosome: dict, ant_layers: int,config:Config):
    weights = {}
    transmitting_neruons = dict(filter(lambda keyValue: keyValue[1].parent_layer == 0, neuron_chromosome.items()))
    for receiving_layer in range(1, ant_layers):
        receiving_neruons = dict(
            filter(lambda keyValue: keyValue[1].parent_layer == receiving_layer, neuron_chromosome.items()))
        for transmitting_neuron in transmitting_neruons:
            for receiving_neuron in receiving_neruons:  # for now fully connected to adjesent layers
                id = next(config.global_weight_id_counter)
                weights[id] = WeightGene(transmitting_neuron=transmitting_neuron,
                                         receiving_neuron=receiving_neuron,
                                         innovationNumber=id,
                                         value=random.uniform(config.genome_value_min, config.genome_value_max))

        transmitting_neruons = receiving_neruons
    return weights
