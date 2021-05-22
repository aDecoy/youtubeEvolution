import gym
import numpy
import random
from statistics import mean, stdev, pstdev
from itertools import count
from copy import deepcopy
from collections import Counter
import time
from dataclasses import dataclass

global_individ_id_counter = count()
global_bias_id_counter = count()
global_weight_id_counter = count()


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

    def mutate(self):
        pass


@dataclass
class NeuronGene:
    """Class for keeping track of an gene"""
    innovationNumber: int = None
    parent_layer: str = None
    can_average_values_in_crossover = False

    def mutate(self):
        pass


@dataclass
class BiasGene:
    """Class for keeping track of an gene"""
    value: float
    parent_neuron: int = None
    innovationNumber: int = None
    can_average_values_in_crossover = True

    def mutate(self):
        floatValueMutation(self)


@dataclass
class WeightGene:
    """Class for keeping track of an gene"""
    value: float
    innovationNumber: int = None
    transmitting_neuron: str = None
    receiving_neuron: str = None
    can_average_values_in_crossover = True

    def mutate(self):
        floatValueMutation(self)


@dataclass
class Phenotype:
    """Weights are  layer1-layer2 : 2d list of weights for matrix multiplication"""
    """Biases are  layer : 1d list of biases"""
    weights: dict
    biases: dict


def floatValueMutation(gene: Gene):
    if random.random() < mutation_rate:
        # Mutate
        gene.value += random.uniform(-mutation_power, +mutation_power)


class NeuralNetworkModel:
    def __init__(self, phenotype: Phenotype = None, id=None):

        self.weights = phenotype.weights

        # self.biases = numpy.array(genome["biases"])
        self.biases = phenotype.biases

    def noise_output(self):
        return random.choice(range(3))

    def run(self, input_values):
        if use_action_noise:
            if random.random() < action_noise_rate:
                return self.noise_output()

        layer_values = [input_values]
        for layer in range(1, len(layer_sizes)):
            incoming_values = layer_values[-1]
            neruon_sums = numpy.matmul(self.weights["{}-{}".format(layer - 1, layer)], incoming_values) + self.biases[layer]
            layer_values.append(neruon_sums)



        # action_choise = numpy.argmax(layer_values[-1])
        action_choise = layer_values[-1]
        # action_choise = numpy.max(layer_sizes[-1], (1))

        return action_choise

        # https://www.youtube.com/watch?v=lFOOjeH2wsY
        # https://www.youtube.com/watch?v=woa34ugDSwY
        # https://www.youtube.com/watch?v=wasZ0MusbdM&list=PL_mqLx7AmDzeG5kXYbhllIaLiZIALla3P&index=5
        # todo clamp / normalize outputs  with activation function


def evaluate(individual):
    total_score = []
    # perceptron = NeuralNetworkModel(individual["genome"])
    phenotype = develop_genome_to_neural_network_phenotype(individual["genome"])  # todo mix these ? change function  name?
    # phenotype = develop_genome_to_neural_network_phenotype(geneBcontinious["genome"])
    nerualNetwork = NeuralNetworkModel(phenotype)
    # perceptron = NeuralNetworkModel(gene["genome"])
    for i_episode in range(ant_simulations):
        observation = env.reset()
        score_in_current_simulation = 0
        for t in range(simulation_max_steps):
            # env.render()
            # print(observation)
            # action = env.action_space.sample()
            # print(action)
            if use_observation_noise:
                observation = noizify_observations(observation)
            action = nerualNetwork.run(observation)
            observation, reward, done, info = env.step(action)
            score_in_current_simulation += reward
            if done:
                # print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation ))
                total_score.append(score_in_current_simulation)
                break

    fitness = sum(total_score) / ant_simulations
    # fitness = min(total_score)
    individual["fitness"] = fitness
    # print("Average score for individual {}:  {}".format(individual["id"],fitness))

    return fitness


def create_new_individual():
    # genome = {"weights": [1, random.random(), 1, random.random(), 0, 1, random.random(), 1],
    #           "biases": [0, random.random(), 0, 0, 2, random.random(), 0, 0, ]}
    id = next(global_individ_id_counter)
    genome = create_new_genome(id, layer_sizes)
    individual = {"id": id, "genome": genome}
    return individual


# todo input har - verdier fordi at det ikke gir noe mening med ulike innovasjonsnummere for IO neruoner. Output har enten veldig høyer , eller 0 til ant output

# todo individuals all have the same id numbers for hte  hidden nodes all have the same , to make it easier (more iterative). before using neat
#
def create_new_genome(id, layer_sizes: list):
    layers = create_new_layer_chromosome(ant_layers=len(layer_sizes))
    neurons = create_new_neruon_chromosome(layer_sizes=layer_sizes)
    biases = create_new_bias_chromosome(neurons)
    weights = create_new_weight_chromosome(neuron_chromosome=neurons, ant_layers=len(layer_sizes))
    genome = Genome(
        layer_chromosome=layers,
        neuron_chromosome=neurons,
        bias_chromosome=biases,
        weight_chromosome=weights)
    return genome


def create_new_layer_chromosome(ant_layers):
    return {i: LayerGene(level=i) for i in range(ant_layers)}


def create_new_neruon_chromosome(layer_sizes):
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
            id = next(global_neruon_id_counter)
            neruons[id] = NeuronGene(innovationNumber=id, parent_layer=layer)
    # Output Neruons
    output_innovation_number = 0
    for i in range(layer_sizes[-1]):
        id = output_innovation_number
        output_innovation_number += 1
        neruons[id] = NeuronGene(innovationNumber=id, parent_layer=len(layer_sizes) - 1)

    return neruons


def create_new_bias_chromosome(neuron_chromosome):
    biases = {}
    for neuron_id in neuron_chromosome:
        if neuron_id >= 0:
            biases[neuron_id] = BiasGene(parent_neuron=neuron_id, value=random.uniform(genome_value_min, genome_value_max))
    return biases


def create_new_weight_chromosome(neuron_chromosome: dict, ant_layers: int):
    weights = {}
    transmitting_neruons = dict(filter(lambda keyValue: keyValue[1].parent_layer == 0, neuron_chromosome.items()))
    for receiving_layer in range(1, ant_layers):
        receiving_neruons = dict(filter(lambda keyValue: keyValue[1].parent_layer == receiving_layer, neuron_chromosome.items()))
        for transmitting_neuron in transmitting_neruons:
            for receiving_neuron in receiving_neruons:  # for now fully connected to adjesent layers
                id = next(global_weight_id_counter)
                weights[id] = WeightGene(transmitting_neuron=transmitting_neuron,
                                         receiving_neuron=receiving_neuron,
                                         innovationNumber=id,
                                         value=random.uniform(genome_value_min, genome_value_max))

        transmitting_neruons = receiving_neruons
    return weights


def develop_genome_to_neural_network_phenotype(genome: Genome):
    ant_layers = len(genome.layer_chromosome)
    biases = {}
    for receiving_layer in range(1, ant_layers):
        receiving_neruons = [key for (key, neuron) in genome.neuron_chromosome.items() if neuron.parent_layer == receiving_layer]
        # dont need to care about matching weight and bias order yet, since no noeruans are added or removed at the moment
        biases[receiving_layer] = numpy.array([genome.bias_chromosome[key].value for key in receiving_neruons])

    weights = {}
    transmitting_neruons = [key for (key, neuron) in genome.neuron_chromosome.items() if neuron.parent_layer == 0]
    for receiving_layer in range(1, ant_layers):
        receiving_neruons = [key for (key, neuron) in genome.neuron_chromosome.items() if neuron.parent_layer == receiving_layer]
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


def initialize_populatin(pop_size):
    population = []
    for _ in range(pop_size):
        population.append(create_new_individual())
    return population


def replenish_population_with_new_random(pop_size, population):
    individuals_killed = pop_size - len(population)
    for _ in range(individuals_killed):
        individ = create_new_individual()
        population.append(individ)

    return population


def mutate(individual):
    genome = individual["genome"]
    for chromosome_key, chromosome in vars(genome).items():
        # new_chromosome = []
        # if isinstance(genome.chromosomes[chromosome], list):
        for key in chromosome:
            chromosome[key].mutate()


def crossover(parent1, parent2):
    assert parent1 != parent2
    # Setup

    if parent1["fitness"] < parent2["fitness"]:
        parent1, parent2 = deepcopy(parent2), deepcopy(parent1)

    p1_genome = parent1["genome"]
    p2_genome = parent1["genome"]
    id = next(global_individ_id_counter)
    child = {"id": id, "genome": Genome()}
    # Mix the genes
    chromosmes_that_can_mix_values = ["weight_chromosome", "bias_chromosome"]

    # crossover_neruons
    for chromosome_key, parent1_chromsome in vars(p1_genome).items():
        parent2_chromsome = p2_genome.get(chromosome_key)
        child_chromsome = deepcopy(parent1_chromsome)
        # Cross over chromoome
        # for gene_id in range(len(child_chromsome)):
        for id in child_chromsome:
            # parent1_gene_list = child_chromsome[id]
            # parent2_gene_list = parent2_chromsome[id]
            # for i, p1_gene in enumerate(parent1_gene_list):
            if id in parent2_chromsome:
                crossover_event = random.random()
                if child_chromsome[id].can_average_values_in_crossover:
                    if chromosome_key in chromosmes_that_can_mix_values and crossover_event < crossover_mix_genes_rate:
                        child_chromsome[id].value = (parent1_chromsome[id].value + parent2_chromsome[id].value) * 0.5
                    elif crossover_event < crossover_mix_genes_rate + (1 - crossover_mix_genes_rate) / 2:
                        child_chromsome[id] = parent2_chromsome[id]
                    ##else keep parent 1
                else:
                    if crossover_event < 0.5:
                        child_chromsome[id] = parent2_chromsome[id]
                    ##else keep parent 1

        child["genome"].__setattr__(chromosome_key, child_chromsome)
    return child


def noizify_observations(observations):
    max_observation_noise = 0.2
    noise_mean = 0
    noise = numpy.random.normal(loc=noise_mean, scale=observation_noise_std, size=8)
    observations = observations + noise
    return observations


if __name__ == '__main__':
    # todo : look for bottle necks

    env = gym.make('LunarLanderContinuous-v2')
    ## Configuration values
    # Genome config
    genome_value_min = -1
    genome_value_max = 1
    # ant_hidden_layers = 0
    ant_input_neruons = 8
    ant_output_neruons = 2
    layer_sizes = [ant_input_neruons, 6, 4, ant_output_neruons]
    # Simulation run config
    simulation_max_steps = 350
    generations = 30
    pop_size = 150
    # ant_simulations = 15
    ant_simulations = 30
    observation_noise_std = 0.05
    use_observation_noise = False
    use_action_noise = False
    action_noise_rate = 0.01
    # GA params
    mutation_rate = 0.05
    mutation_power = 0.4
    crossover_mix_genes_rate = 0.1
    parent_selection_weighted_choises = False

    global_neruon_id_counter = count(start=ant_output_neruons)

    ## Init population
    population = initialize_populatin(pop_size)

    # geneB =  {'id': 2219, 'genome': Genome(layer_chromosome={0: LayerGene(level=0, innovationNumber=None, activation_function=None), 1: LayerGene(level=1, innovationNumber=None, activation_function=None), 2: LayerGene(level=2, innovationNumber=None, activation_function=None)}, neuron_chromosome={-1: NeuronGene(innovationNumber=-1, parent_layer=0), -2: NeuronGene(innovationNumber=-2, parent_layer=0), -3: NeuronGene(innovationNumber=-3, parent_layer=0), -4: NeuronGene(innovationNumber=-4, parent_layer=0), -5: NeuronGene(innovationNumber=-5, parent_layer=0), -6: NeuronGene(innovationNumber=-6, parent_layer=0), -7: NeuronGene(innovationNumber=-7, parent_layer=0), -8: NeuronGene(innovationNumber=-8, parent_layer=0), 108: NeuronGene(innovationNumber=108, parent_layer=1), 109: NeuronGene(innovationNumber=109, parent_layer=1), 110: NeuronGene(innovationNumber=110, parent_layer=1), 111: NeuronGene(innovationNumber=111, parent_layer=1), 0: NeuronGene(innovationNumber=0, parent_layer=2), 1: NeuronGene(innovationNumber=1, parent_layer=2), 2: NeuronGene(innovationNumber=2, parent_layer=2), 3: NeuronGene(innovationNumber=3, parent_layer=2)}, weight_chromosome={1248: WeightGene(value=0.9433276604105858, innovationNumber=1248, transmitting_neuron=-1, receiving_neuron=108), 1249: WeightGene(value=1.3309991359146838, innovationNumber=1249, transmitting_neuron=-1, receiving_neuron=109), 1250: WeightGene(value=0.8296905620249604, innovationNumber=1250, transmitting_neuron=-1, receiving_neuron=110), 1251: WeightGene(value=0.6173854220434225, innovationNumber=1251, transmitting_neuron=-1, receiving_neuron=111), 1252: WeightGene(value=1.2315522142001747, innovationNumber=1252, transmitting_neuron=-2, receiving_neuron=108), 1253: WeightGene(value=0.1717471046074385, innovationNumber=1253, transmitting_neuron=-2, receiving_neuron=109), 1254: WeightGene(value=-0.8730469043090361, innovationNumber=1254, transmitting_neuron=-2, receiving_neuron=110), 1255: WeightGene(value=0.14062688160851422, innovationNumber=1255, transmitting_neuron=-2, receiving_neuron=111), 1256: WeightGene(value=0.09762408450695048, innovationNumber=1256, transmitting_neuron=-3, receiving_neuron=108), 1257: WeightGene(value=0.19577176253156636, innovationNumber=1257, transmitting_neuron=-3, receiving_neuron=109), 1258: WeightGene(value=-0.15640976446709376, innovationNumber=1258, transmitting_neuron=-3, receiving_neuron=110), 1259: WeightGene(value=-0.0463296916910145, innovationNumber=1259, transmitting_neuron=-3, receiving_neuron=111), 1260: WeightGene(value=0.4322197991572294, innovationNumber=1260, transmitting_neuron=-4, receiving_neuron=108), 1261: WeightGene(value=-0.4405215085801435, innovationNumber=1261, transmitting_neuron=-4, receiving_neuron=109), 1262: WeightGene(value=-0.6708439704209194, innovationNumber=1262, transmitting_neuron=-4, receiving_neuron=110), 1263: WeightGene(value=1.2137582682183132, innovationNumber=1263, transmitting_neuron=-4, receiving_neuron=111), 1264: WeightGene(value=-1.2051739653304212, innovationNumber=1264, transmitting_neuron=-5, receiving_neuron=108), 1265: WeightGene(value=-0.38875891149916797, innovationNumber=1265, transmitting_neuron=-5, receiving_neuron=109), 1266: WeightGene(value=-0.006313907304450916, innovationNumber=1266, transmitting_neuron=-5, receiving_neuron=110), 1267: WeightGene(value=-0.5561866113517258, innovationNumber=1267, transmitting_neuron=-5, receiving_neuron=111), 1268: WeightGene(value=0.3031599686327767, innovationNumber=1268, transmitting_neuron=-6, receiving_neuron=108), 1269: WeightGene(value=-0.8095897889960313, innovationNumber=1269, transmitting_neuron=-6, receiving_neuron=109), 1270: WeightGene(value=0.1586314698285315, innovationNumber=1270, transmitting_neuron=-6, receiving_neuron=110), 1271: WeightGene(value=0.9918726184859533, innovationNumber=1271, transmitting_neuron=-6, receiving_neuron=111), 1272: WeightGene(value=-0.7355105369662676, innovationNumber=1272, transmitting_neuron=-7, receiving_neuron=108), 1273: WeightGene(value=-0.3899354078985431, innovationNumber=1273, transmitting_neuron=-7, receiving_neuron=109), 1274: WeightGene(value=-0.4394729613030534, innovationNumber=1274, transmitting_neuron=-7, receiving_neuron=110), 1275: WeightGene(value=0.22655646691765957, innovationNumber=1275, transmitting_neuron=-7, receiving_neuron=111), 1276: WeightGene(value=0.8268646338432031, innovationNumber=1276, transmitting_neuron=-8, receiving_neuron=108), 1277: WeightGene(value=-0.848737706344886, innovationNumber=1277, transmitting_neuron=-8, receiving_neuron=109), 1278: WeightGene(value=-0.9770976777158128, innovationNumber=1278, transmitting_neuron=-8, receiving_neuron=110), 1279: WeightGene(value=0.30049549592053476, innovationNumber=1279, transmitting_neuron=-8, receiving_neuron=111), 1280: WeightGene(value=0.7395496008902067, innovationNumber=1280, transmitting_neuron=108, receiving_neuron=0), 1281: WeightGene(value=-0.6690782608690221, innovationNumber=1281, transmitting_neuron=108, receiving_neuron=1), 1282: WeightGene(value=-0.08775073763817653, innovationNumber=1282, transmitting_neuron=108, receiving_neuron=2), 1283: WeightGene(value=0.14110829925703278, innovationNumber=1283, transmitting_neuron=108, receiving_neuron=3), 1284: WeightGene(value=0.6927498596772106, innovationNumber=1284, transmitting_neuron=109, receiving_neuron=0), 1285: WeightGene(value=0.17024718669516548, innovationNumber=1285, transmitting_neuron=109, receiving_neuron=1), 1286: WeightGene(value=0.7420756748955472, innovationNumber=1286, transmitting_neuron=109, receiving_neuron=2), 1287: WeightGene(value=0.3721029843758649, innovationNumber=1287, transmitting_neuron=109, receiving_neuron=3), 1288: WeightGene(value=-1.1755119055162369, innovationNumber=1288, transmitting_neuron=110, receiving_neuron=0), 1289: WeightGene(value=0.16905377466895763, innovationNumber=1289, transmitting_neuron=110, receiving_neuron=1), 1290: WeightGene(value=0.012035969476737685, innovationNumber=1290, transmitting_neuron=110, receiving_neuron=2), 1291: WeightGene(value=-0.4754311329467581, innovationNumber=1291, transmitting_neuron=110, receiving_neuron=3), 1292: WeightGene(value=1.1958463640363421, innovationNumber=1292, transmitting_neuron=111, receiving_neuron=0), 1293: WeightGene(value=0.043463871930397635, innovationNumber=1293, transmitting_neuron=111, receiving_neuron=1), 1294: WeightGene(value=0.026669355679097284, innovationNumber=1294, transmitting_neuron=111, receiving_neuron=2), 1295: WeightGene(value=-0.03835137731020505, innovationNumber=1295, transmitting_neuron=111, receiving_neuron=3)}, bias_chromosome={108: BiasGene(value=0.7668089188110419, parent_neuron=108, innovationNumber=None), 109: BiasGene(value=0.6232083939584674, parent_neuron=109, innovationNumber=None), 110: BiasGene(value=0.1525173479877075, parent_neuron=110, innovationNumber=None), 111: BiasGene(value=0.38175093124723825, parent_neuron=111, innovationNumber=None), 0: BiasGene(value=0.46697282789873307, parent_neuron=0, innovationNumber=None), 1: BiasGene(value=0.49660603710994705, parent_neuron=1, innovationNumber=None), 2: BiasGene(value=0.6718369588691699, parent_neuron=2, innovationNumber=None), 3: BiasGene(value=0.736153780701598, parent_neuron=3, innovationNumber=None)}), 'fitness': -8.393776440787112}

    geneBcontinious = {'id': 3813, 'genome': Genome(layer_chromosome={0: LayerGene(level=0, innovationNumber=None, activation_function=None), 1: LayerGene(level=1, innovationNumber=None, activation_function=None), 2: LayerGene(level=2, innovationNumber=None, activation_function=None), 3: LayerGene(level=3, innovationNumber=None, activation_function=None)}, neuron_chromosome={-1: NeuronGene(innovationNumber=-1, parent_layer=0), -2: NeuronGene(innovationNumber=-2, parent_layer=0), -3: NeuronGene(innovationNumber=-3, parent_layer=0), -4: NeuronGene(innovationNumber=-4, parent_layer=0), -5: NeuronGene(innovationNumber=-5, parent_layer=0), -6: NeuronGene(innovationNumber=-6, parent_layer=0), -7: NeuronGene(innovationNumber=-7, parent_layer=0), -8: NeuronGene(innovationNumber=-8, parent_layer=0), 852: NeuronGene(innovationNumber=852, parent_layer=1), 853: NeuronGene(innovationNumber=853, parent_layer=1), 854: NeuronGene(innovationNumber=854, parent_layer=1), 855: NeuronGene(innovationNumber=855, parent_layer=1), 856: NeuronGene(innovationNumber=856, parent_layer=1), 857: NeuronGene(innovationNumber=857, parent_layer=1), 858: NeuronGene(innovationNumber=858, parent_layer=2), 859: NeuronGene(innovationNumber=859, parent_layer=2), 860: NeuronGene(innovationNumber=860, parent_layer=2), 861: NeuronGene(innovationNumber=861, parent_layer=2), 0: NeuronGene(innovationNumber=0, parent_layer=3), 1: NeuronGene(innovationNumber=1, parent_layer=3)}, weight_chromosome={6800: WeightGene(value=0.8419477584511708, innovationNumber=6800, transmitting_neuron=-1, receiving_neuron=852), 6801: WeightGene(value=0.0594608760893478, innovationNumber=6801, transmitting_neuron=-1, receiving_neuron=853), 6802: WeightGene(value=-0.3275509785655307, innovationNumber=6802, transmitting_neuron=-1, receiving_neuron=854), 6803: WeightGene(value=1.0076745950221588, innovationNumber=6803, transmitting_neuron=-1, receiving_neuron=855), 6804: WeightGene(value=-0.5623174058174644, innovationNumber=6804, transmitting_neuron=-1, receiving_neuron=856), 6805: WeightGene(value=0.724876776977124, innovationNumber=6805, transmitting_neuron=-1, receiving_neuron=857), 6806: WeightGene(value=0.24997944298957508, innovationNumber=6806, transmitting_neuron=-2, receiving_neuron=852), 6807: WeightGene(value=-0.43010606464643303, innovationNumber=6807, transmitting_neuron=-2, receiving_neuron=853), 6808: WeightGene(value=0.4584642506408606, innovationNumber=6808, transmitting_neuron=-2, receiving_neuron=854), 6809: WeightGene(value=0.8546602770115992, innovationNumber=6809, transmitting_neuron=-2, receiving_neuron=855), 6810: WeightGene(value=1.2225718493025735, innovationNumber=6810, transmitting_neuron=-2, receiving_neuron=856), 6811: WeightGene(value=-0.07789586612223964, innovationNumber=6811, transmitting_neuron=-2, receiving_neuron=857), 6812: WeightGene(value=-0.13425106272758064, innovationNumber=6812, transmitting_neuron=-3, receiving_neuron=852), 6813: WeightGene(value=-0.08759487599719665, innovationNumber=6813, transmitting_neuron=-3, receiving_neuron=853), 6814: WeightGene(value=0.1870749371487011, innovationNumber=6814, transmitting_neuron=-3, receiving_neuron=854), 6815: WeightGene(value=0.5107529467974726, innovationNumber=6815, transmitting_neuron=-3, receiving_neuron=855), 6816: WeightGene(value=-0.9511873630608585, innovationNumber=6816, transmitting_neuron=-3, receiving_neuron=856), 6817: WeightGene(value=0.9432863937544782, innovationNumber=6817, transmitting_neuron=-3, receiving_neuron=857), 6818: WeightGene(value=-1.2453814929550022, innovationNumber=6818, transmitting_neuron=-4, receiving_neuron=852), 6819: WeightGene(value=0.3523378508966101, innovationNumber=6819, transmitting_neuron=-4, receiving_neuron=853), 6820: WeightGene(value=0.6529622819985678, innovationNumber=6820, transmitting_neuron=-4, receiving_neuron=854), 6821: WeightGene(value=0.7166723775444797, innovationNumber=6821, transmitting_neuron=-4, receiving_neuron=855), 6822: WeightGene(value=0.41695105821702005, innovationNumber=6822, transmitting_neuron=-4, receiving_neuron=856), 6823: WeightGene(value=-0.060531298896785324, innovationNumber=6823, transmitting_neuron=-4, receiving_neuron=857), 6824: WeightGene(value=-0.12137713495419175, innovationNumber=6824, transmitting_neuron=-5, receiving_neuron=852), 6825: WeightGene(value=1.405310037004512, innovationNumber=6825, transmitting_neuron=-5, receiving_neuron=853), 6826: WeightGene(value=0.39268508973759586, innovationNumber=6826, transmitting_neuron=-5, receiving_neuron=854), 6827: WeightGene(value=-1.651762039753849, innovationNumber=6827, transmitting_neuron=-5, receiving_neuron=855), 6828: WeightGene(value=0.7884206370119518, innovationNumber=6828, transmitting_neuron=-5, receiving_neuron=856), 6829: WeightGene(value=0.5590969619862274, innovationNumber=6829, transmitting_neuron=-5, receiving_neuron=857), 6830: WeightGene(value=-0.02333805608298739, innovationNumber=6830, transmitting_neuron=-6, receiving_neuron=852), 6831: WeightGene(value=-0.3672745463966809, innovationNumber=6831, transmitting_neuron=-6, receiving_neuron=853), 6832: WeightGene(value=0.3271771463406697, innovationNumber=6832, transmitting_neuron=-6, receiving_neuron=854), 6833: WeightGene(value=-0.9076365767251157, innovationNumber=6833, transmitting_neuron=-6, receiving_neuron=855), 6834: WeightGene(value=-0.06278422227690222, innovationNumber=6834, transmitting_neuron=-6, receiving_neuron=856), 6835: WeightGene(value=0.773225546207263, innovationNumber=6835, transmitting_neuron=-6, receiving_neuron=857), 6836: WeightGene(value=-1.1213425675869377, innovationNumber=6836, transmitting_neuron=-7, receiving_neuron=852), 6837: WeightGene(value=0.8231232303940725, innovationNumber=6837, transmitting_neuron=-7, receiving_neuron=853), 6838: WeightGene(value=-0.2215456416786648, innovationNumber=6838, transmitting_neuron=-7, receiving_neuron=854), 6839: WeightGene(value=1.206431664558393, innovationNumber=6839, transmitting_neuron=-7, receiving_neuron=855), 6840: WeightGene(value=0.4454429406274577, innovationNumber=6840, transmitting_neuron=-7, receiving_neuron=856), 6841: WeightGene(value=0.4516003402366456, innovationNumber=6841, transmitting_neuron=-7, receiving_neuron=857), 6842: WeightGene(value=-1.1126818926215658, innovationNumber=6842, transmitting_neuron=-8, receiving_neuron=852), 6843: WeightGene(value=-0.8943846773693165, innovationNumber=6843, transmitting_neuron=-8, receiving_neuron=853), 6844: WeightGene(value=0.027620256545646482, innovationNumber=6844, transmitting_neuron=-8, receiving_neuron=854), 6845: WeightGene(value=-1.0274070539953597, innovationNumber=6845, transmitting_neuron=-8, receiving_neuron=855), 6846: WeightGene(value=0.9699739270438552, innovationNumber=6846, transmitting_neuron=-8, receiving_neuron=856), 6847: WeightGene(value=0.09386229096931731, innovationNumber=6847, transmitting_neuron=-8, receiving_neuron=857), 6848: WeightGene(value=-0.04629673898989073, innovationNumber=6848, transmitting_neuron=852, receiving_neuron=858), 6849: WeightGene(value=0.05242501057049731, innovationNumber=6849, transmitting_neuron=852, receiving_neuron=859), 6850: WeightGene(value=-0.03893587738079052, innovationNumber=6850, transmitting_neuron=852, receiving_neuron=860), 6851: WeightGene(value=-0.5868031004053025, innovationNumber=6851, transmitting_neuron=852, receiving_neuron=861), 6852: WeightGene(value=-0.16835899558917455, innovationNumber=6852, transmitting_neuron=853, receiving_neuron=858), 6853: WeightGene(value=-1.0961098892833696, innovationNumber=6853, transmitting_neuron=853, receiving_neuron=859), 6854: WeightGene(value=-0.7205280712423877, innovationNumber=6854, transmitting_neuron=853, receiving_neuron=860), 6855: WeightGene(value=0.9385577796415341, innovationNumber=6855, transmitting_neuron=853, receiving_neuron=861), 6856: WeightGene(value=-1.1612255613394298, innovationNumber=6856, transmitting_neuron=854, receiving_neuron=858), 6857: WeightGene(value=-0.2932644089474665, innovationNumber=6857, transmitting_neuron=854, receiving_neuron=859), 6858: WeightGene(value=-0.8423816841184053, innovationNumber=6858, transmitting_neuron=854, receiving_neuron=860), 6859: WeightGene(value=0.6929832140921265, innovationNumber=6859, transmitting_neuron=854, receiving_neuron=861), 6860: WeightGene(value=-1.0759419070888914, innovationNumber=6860, transmitting_neuron=855, receiving_neuron=858), 6861: WeightGene(value=0.47074491332121526, innovationNumber=6861, transmitting_neuron=855, receiving_neuron=859), 6862: WeightGene(value=0.7909771638281186, innovationNumber=6862, transmitting_neuron=855, receiving_neuron=860), 6863: WeightGene(value=1.0613181556468736, innovationNumber=6863, transmitting_neuron=855, receiving_neuron=861), 6864: WeightGene(value=1.0036069006465216, innovationNumber=6864, transmitting_neuron=856, receiving_neuron=858), 6865: WeightGene(value=-0.6166741345525393, innovationNumber=6865, transmitting_neuron=856, receiving_neuron=859), 6866: WeightGene(value=0.0397132342664982, innovationNumber=6866, transmitting_neuron=856, receiving_neuron=860), 6867: WeightGene(value=0.4602718667667426, innovationNumber=6867, transmitting_neuron=856, receiving_neuron=861), 6868: WeightGene(value=0.6118371099736353, innovationNumber=6868, transmitting_neuron=857, receiving_neuron=858), 6869: WeightGene(value=0.4188615266766731, innovationNumber=6869, transmitting_neuron=857, receiving_neuron=859), 6870: WeightGene(value=0.40100324690272837, innovationNumber=6870, transmitting_neuron=857, receiving_neuron=860), 6871: WeightGene(value=-0.40850513898646895, innovationNumber=6871, transmitting_neuron=857, receiving_neuron=861), 6872: WeightGene(value=-0.3483325850316768, innovationNumber=6872, transmitting_neuron=858, receiving_neuron=0), 6873: WeightGene(value=0.5404994930353584, innovationNumber=6873, transmitting_neuron=858, receiving_neuron=1), 6874: WeightGene(value=0.6040180568118126, innovationNumber=6874, transmitting_neuron=859, receiving_neuron=0), 6875: WeightGene(value=-0.4759249106685792, innovationNumber=6875, transmitting_neuron=859, receiving_neuron=1), 6876: WeightGene(value=-0.2596557988750514, innovationNumber=6876, transmitting_neuron=860, receiving_neuron=0), 6877: WeightGene(value=-0.7768641893574211, innovationNumber=6877, transmitting_neuron=860, receiving_neuron=1), 6878: WeightGene(value=-1.938478514291704, innovationNumber=6878, transmitting_neuron=861, receiving_neuron=0), 6879: WeightGene(value=-0.44644067878366195, innovationNumber=6879, transmitting_neuron=861, receiving_neuron=1)}, bias_chromosome={852: BiasGene(value=-0.011993802012837507, parent_neuron=852, innovationNumber=None), 853: BiasGene(value=-0.12076139949312967, parent_neuron=853, innovationNumber=None), 854: BiasGene(value=0.4223137549264695, parent_neuron=854, innovationNumber=None), 855: BiasGene(value=0.41953570379175165, parent_neuron=855, innovationNumber=None), 856: BiasGene(value=-0.5090314359536778, parent_neuron=856, innovationNumber=None), 857: BiasGene(value=-0.40903044409179423, parent_neuron=857, innovationNumber=None), 858: BiasGene(value=0.36525348927631385, parent_neuron=858, innovationNumber=None), 859: BiasGene(value=-0.04881752108623205, parent_neuron=859, innovationNumber=None), 860: BiasGene(value=-0.44514307995923347, parent_neuron=860, innovationNumber=None), 861: BiasGene(value=-0.25123275041610216, parent_neuron=861, innovationNumber=None), 0: BiasGene(value=-0.14854826152858291, parent_neuron=0, innovationNumber=None), 1: BiasGene(value=0.5340888591242442, parent_neuron=1, innovationNumber=None)}), 'fitness': 262.0968771826852}


    # layer_sizes = [ant_input_neruons, 4, ant_output_neruons]

    for g in range(generations):
        # print("Generation {}".format(g))
        ## Evaualte
        for individual in population:
            evaluate(individual)

        after_evaluation = time.time()
        ## Summarize current generation
        best = max(population, key=lambda individual: individual["fitness"])
        average_population_fitness = mean([individual["fitness"] for individual in population])
        print("Best Individual in generation {}: id:{}, fitness {}".format(g, best["id"], best["fitness"]))
        print("Populatation average in generation {} was {}".format(g, average_population_fitness))

        print('------------------------------')

        # Set up population for next generation
        if g != generations - 1:
            population.sort(key=lambda individual: individual["fitness"], reverse=True)

            # Survival selection
            survival_rate = 0.10
            ant_survivors = round(len(population) * survival_rate)
            survivers = population[0:ant_survivors]

            # random_new_rate = 0.00
            # ant_random_new = round(len(population) * random_new_rate)
            ant_random_new = 0

            # Potential_Parent selection
            parent_rate = 0.10
            ant_parents = max(round(parent_rate * pop_size), 2)
            potential_parents = population[0:ant_parents]

            if parent_selection_weighted_choises:
                parent_fitnesses = [x["fitness"] for x in potential_parents]
                parent_fitnesses_range = abs(parent_fitnesses[0] - parent_fitnesses[-1])
                normalize_fitness_performance = [round(abs(fitness - parent_fitnesses[-1])) / parent_fitnesses_range for fitness in
                                                 parent_fitnesses]
                sumOfP = sum(normalize_fitness_performance)
                reproduction_probability = [p / sumOfP for p in normalize_fitness_performance]

            ant_children = len(population) - ant_survivors - ant_random_new
            children = []
            for i in range(ant_children):
                # parent = potential_parents[random.randrange(0, len(potential_parents))]
                # Parent selection
                if parent_selection_weighted_choises:
                    parent1, parent2 = numpy.random.choice(potential_parents, size=2, p=reproduction_probability, replace=False)
                else:
                    parent1, parent2 = random.sample(potential_parents, k=2)

                child = crossover(parent1, parent2)
                mutate(child)
                children.append(child)

            population = survivers + children

            # replenish_population_with_new_random(pop_size, population)

            assert len(population) == pop_size

            non_evalutation_time = time.time() - after_evaluation
            print(non_evalutation_time)

    ### Summary
    best = max(population, key=lambda individual: individual["fitness"])
    print("best Individual got score {}, and looked like:   {}".format(best["fitness"], best))
    average_population_fitness = mean([individual["fitness"] for individual in population])
    print("average population fitness:   {}".format(average_population_fitness))

    env.close()
