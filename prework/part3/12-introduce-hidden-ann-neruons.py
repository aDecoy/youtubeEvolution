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
    # todo Genome representation
    # todo option 1 list: [[weights-1, biases-1][weights-2,biases-2]]
    # todo option 1 set: {1: [weights-1, biases-1], 2: [weights-2,biases-2]}
    # todo option 2: [[weights-1, weights-2][biases-1, biases-2]]
    # todo option 2: {weights:[weights-1, weights-2], biases: [biases-1, biases-2]}  <-

    # todo option 2 set set: {weights:{1:weights-1, 2: weights-2}, biases: {1: biases-1, 2: biases-2}}

    # todo Not options: 8 neruon input, 1 neruon output.
    # todo Is an option: everything in between

    # def __init__(self, genome={"weights": [2, 0.9, 1.2, 1.1, 1, 1, 1, 1], "biases": [0, 0, 0, 0, 0, 0, 0, 0, ]}):
    # def __init__(self, genome={"weights":[[1,1, 1, 1, 1, 1, 1, 1],[1,1,1,1]], "biases": [[0, 0, 0, 0, 0, 0, 0, 0 ],[0,0,0,0]]}):
    def __init__(self, phenotype: Phenotype = None, id=None):
        # if genome is None:
        #     genome = {"weights": {"0-1": [[1, 1, 1, 1, 1, 1, 1, 1],
        #                                   [1, 1, 1, 1, 1, 1, 1, 1],
        #                                   [1, 1, 1, 1, 1, 1, 1, 1],
        #                                   [1, 1, 1, 1, 1, 1, 1, 1]],
        #                           "1-2": [[1, 1, 1, 1],
        #                                   [1, 1, 1, 1],
        #                                   [1, 1, 1, 1],
        #                                   [1, 1, 1, 1]]},
        #               "biases": [[0, 0, 0, 0], [0]]}
        # if genome is None:
        #     genome = Genome(id=None,
        #                     chromosomes={Chromosome(name="weights",
        #                                             genes={
        #                                                 "0-1": [
        #                                                     [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                     [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                     [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                     [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)]],
        #                                                 "1-2": [[Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1)]]
        #                                             }),
        #                                  Chromosome(name="biases",
        #                                             genes={"1": [Gene(0), Gene(0), Gene(0), Gene(0)], "2": [Gene(0)]})
        #                                  }
        #                     )
        #     if genome is None:
        #         genome = Genome(id=None,
        #                         chromosomes={Chromosome(name="weights",
        #                                                 genes={
        #                                                     "layer_0": [], # here the activation function will been
        #                                                     "layer_1": [],
        #                                                     "layer_2" : [],
        #                                                     "0-1": [
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)]],
        #                                                     "1-2": [[Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                             [Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                             [Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                             [Gene(1), Gene(1), Gene(1), Gene(1)]]
        #                                                 }),
        #                                      Chromosome(name="biases",
        #                                                 genes={"1": [Gene(0), Gene(0), Gene(0), Gene(0)], "2": [Gene(0)]})
        #                                      }
        # #                         )
        #     if genome is None:
        #         genome = Genome(id=None,
        #                         layer_chromosome = {0:LayerGene(level=0 ),1:LayerGene(level=1 ),2:LayerGene(level=2 )}
        #                         neuron_chromosome = {0: NeuronGene(innovationNumber=0, parent_layer=0),
        #                                              1: NeuronGene(innovationNumber=1, parent_layer=0),
        #                                              2: NeuronGene(innovationNumber=2, parent_layer=0),
        #                                              3: NeuronGene(innovationNumber=3, parent_layer=0),
        #                                              4: NeuronGene(innovationNumber=4, parent_layer=0),
        #                                              5: NeuronGene(innovationNumber=5, parent_layer=0),
        #                                              6: NeuronGene(innovationNumber=6, parent_layer=0),
        #                                              7: NeuronGene(innovationNumber=7, parent_layer=0),
        #
        #                                              8: NeuronGene(innovationNumber=8, parent_layer=1),
        #                                              9: NeuronGene(innovationNumber=9, parent_layer=1),
        #                                              10: NeuronGene(innovationNumber=10, parent_layer=1),
        #                                              11: NeuronGene(innovationNumber=11, parent_layer=1),
        #
        #                                              12: NeuronGene(innovationNumber=12, parent_layer=2)
        #                                              },
        #                         bias_chromosome = {0: BiasGene(parent_neuron=0, value=0),
        #                                              1: BiasGene(parent_neuron=1, value=0),
        #                                              2: BiasGene(parent_neuron=2, value=0),
        #                                              3: BiasGene(parent_neuron=3, value=0),
        #                                              4: BiasGene(parent_neuron=4, value=0),
        #                                              5: BiasGene(parent_neuron=5, value=0),
        #                                              6: BiasGene(parent_neuron=6, value=0),
        #                                              7: BiasGene(parent_neuron=7, value=0),
        #
        #                                              8: BiasGene(parent_neuron=8, value=1),
        #                                              9: BiasGene(parent_neuron=9, value=1),
        #                                              10: BiasGene(parent_neuron=10, value=1),
        #                                              11: BiasGene(parent_neuron=11, value=1),
        #
        #                                              12: BiasGene(parent_neuron=12, value=2)
        #                                              },
        #                         chromosomes={Chromosome(name="weights",
        #                                                 genes={
        #                                                     "layer_0": [], # here the activation function will been
        #                                                     "layer_1": [],
        #                                                     "layer_2" : [],
        #                                                     "0-1": [
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                         [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)]],
        #                                                     "1-2": [[Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                             [Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                             [Gene(1), Gene(1), Gene(1), Gene(1)],
        #                                                             [Gene(1), Gene(1), Gene(1), Gene(1)]]
        #                                                 }),
        #                                      Chromosome(name="biases",
        #                                                 genes={"1": [Gene(0), Gene(0), Gene(0), Gene(0)], "2": [Gene(0)]})
        #                                      }
        #                         )
        # self.weights = numpy.array(genome["weights"])
        # self.weights = genome["weights"]
        self.weights = phenotype.weights

        # self.biases = numpy.array(genome["biases"])
        self.biases = phenotype.biases

    def noise_output(self):
        return random.choice(range(3))

    def run(self, input_values):
        if use_action_noise:
            if random.random() < action_noise_rate:
                return self.noise_output()

        # signal_strength = input_values * self.weights + self.biases
        # output_signal = signal_strength.sum()

        # signal_strength_1_to_2_neruon_1 = numpy.matmul(self.weights["0-1"][0], input_values ) + self.biases[0]
        # signal_strength_1_to_2_neruon_2 = numpy.matmul(self.weights["0-1"][1], input_values ) + self.biases[0]

        input_values = numpy.array(input_values).reshape(-1)
        signal_strength_1_to_2 = numpy.matmul(self.weights["0-1"], input_values) + self.biases[1]

        signal_strength_2_to_output = numpy.matmul(self.weights["1-2"], signal_strength_1_to_2) + self.biases[2]
        output_signal = signal_strength_2_to_output.sum()

        # https://www.youtube.com/watch?v=lFOOjeH2wsY
        # https://www.youtube.com/watch?v=woa34ugDSwY
        # https://www.youtube.com/watch?v=wasZ0MusbdM&list=PL_mqLx7AmDzeG5kXYbhllIaLiZIALla3P&index=5
        # todo clamp / normalize outputs  with activation function

        if output_signal < 0.25:
            return 0
        elif output_signal < 0.50:
            return 1
        elif output_signal < 0.75:
            return 2
        else:  # 0.75 to 1nfinity
            return 3

def evaluate(individual):
    total_score = []
    # perceptron = NeuralNetworkModel(individual["genome"])
    phenotype = develop_genome_to_neural_network_phenotype(individual["genome"])  # todo mix these ? change function  name?
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
    # genome = {"weights": {}, "biases": []}
    # for chromosome in genome.chromosomes.keys():
    #
    #     if chromosome == "biases":
    #         for i, ant_neruons in enumerate(layer_sizes[1:],start=1):
    #             # Biases are not added to input layer
    #             new_layer = []
    #             for _ in range(ant_neruons):
    #                 new_layer.append(random.uniform(genome_value_min, genome_value_max))
    #             # genome[chromosome]append(new_layer)
    #             genome.chromosomes["biases"][i]=new_layer
    #
    #     elif chromosome == "weights":
    #         for i, ant_neruons in enumerate(layer_sizes[1:], start=1):
    #             ant_previous_layer_neruons = layer_sizes[i - 1]
    #             weights = numpy.zeros(shape=(ant_neruons, layer_sizes[i - 1]))
    #             ant_weights = ant_neruons * layer_sizes[i - 1]  # fully connected with previous layer
    #             new_layer = []
    #             for _ in range(ant_neruons):
    #                 incoming_signals_to_neuron = []
    #                 for _ in range(ant_previous_layer_neruons):
    #                     incoming_signals_to_neuron.append(random.uniform(genome_value_min, genome_value_max))
    #                 new_layer.append(incoming_signals_to_neuron)
    #             genome.chromosomes["weights"]["{}-{}".format(i - 1, i)] = numpy.array(new_layer)

    # genome = {"weights": [], "biases": []}
    # for chromosome in genome:
    #     for ant_neruons in layer_sizes:
    #         new_layer = []
    #         for _ in range(ant_neruons):
    #             new_layer.append(random.uniform(genome_value_min, genome_value_max))
    #
    #         genome[chromosome].append(new_layer)

    # for chromosome in genome:
    #     # for i in range(8):
    #     for _ in range(ant_hidden_layers):
    #         genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))

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
    for chromosome_key ,chromosome in vars(genome).items():
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

    env = gym.make('LunarLander-v2')
    ## Configuration values
    # Genome config
    genome_value_min = -1
    genome_value_max = 1
    ant_hidden_layers = 0
    ant_input_neruons = 8
    ant_output_neruons = 1
    layer_sizes = [ant_input_neruons, 4, ant_output_neruons]
    # Simulation run config
    simulation_max_steps = 350
    generations = 50
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

    ### Run stuff (evolve the population to find the best individual)
    # gene =  {'id': 34595, 'genome': {'weights': [1.8279610414408418, -1.215151552382231, -1.914386503678685, -1.5992229022420927, -0.4629710592800763, 0.09949141104419473, -0.8159553270867125, 0.3772991566011524], 'biases': [-2.1137517590961803, 0.23877397428813674, 0.6899723535739337, 0.20582457309766966, 1.8034607671721203, -0.9274868121058241, -0.3656953731942547, -0.009541580549235956]}, 'fitness': 16.002701150260478}

    # gene =    {'id': 3989, 'genome': {
    #     'weights': [-0.24589526163185343, -0.9351321015527188, -0.4252530682015365, -0.843237331350113, -0.05330922162596108, -0.28940626568569916,
    #                 -0.5025293210147201, -0.7842060772293385],
    #     'biases': [0.46728910629544984, 0.9255904802474416, -0.19137759905370436, 0.9011485178886629, -0.2366983500766079, 0.027998837730861426,
    #                -0.2321407136431074, -1.2433501315378388]}, 'fitness': -46.83748782227101}

    for g in range(generations):
        # print("Generation {}".format(g))
        ## Evaualte
        for individual in population:
            evaluate(individual)

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
            parent_rate = 0.5
            ant_parents = max(round(parent_rate * pop_size), 2)
            potential_parents = population[0:ant_parents]

            if parent_selection_weighted_choises:
                parent_fitnesses = [x["fitness"] for x in potential_parents]
                parent_fitnesses_range = abs(parent_fitnesses[0] - parent_fitnesses[-1])
                normalize_fitness_performance = [round(abs(fitness - parent_fitnesses[-1])) / parent_fitnesses_range for fitness in parent_fitnesses]
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

            replenish_population_with_new_random(pop_size, population)

            assert len(population) == pop_size

    ### Summary
    best = max(population, key=lambda individual: individual["fitness"])
    print("best Individual got score {}, and looked like:   {}".format(best["fitness"], best))
    average_population_fitness = mean([individual["fitness"] for individual in population])
    print("average population fitness:   {}".format(average_population_fitness))

    env.close()
