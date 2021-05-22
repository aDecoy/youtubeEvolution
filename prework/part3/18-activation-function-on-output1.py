import gym
import numpy
import random
from statistics import mean, stdev, pstdev
from itertools import count
from copy import deepcopy
import time
from dataclasses import dataclass
from os import path, makedirs
import  csv
import json
from dacite import from_dict


global_individ_id_counter = count()
global_bias_id_counter = count()
global_weight_id_counter = count()



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


@dataclass
class Genome:
    """Class for track of an genome with an id from the individual and a dict of chromosomes"""
    # id: "AnyTypeWorkaround" = None # todo trenger ikke denne i genome.
    # chromosomes: Chromosomes = None
    layer_chromosome: dict = None
    neuron_chromosome: dict= None
    weight_chromosome: dict= None
    bias_chromosome: dict= None

    def get(self, chromosome_key):
        return getattr(self, chromosome_key)

@dataclass
class Individ:
    id: int
    genome: Genome
    fitness = None



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


def evaluate(individual: Individ,ant_simulations,  render :bool =False):
    total_score = []
    # perceptron = NeuralNetworkModel(individual.genome)
    phenotype = develop_genome_to_neural_network_phenotype(individual.genome)  # todo mix these ? change function  name?
    # phenotype = develop_genome_to_neural_network_phenotype(geneB.genome)
    nerualNetwork = NeuralNetworkModel(phenotype)
    # perceptron = NeuralNetworkModel(gene.genome)
    for i_episode in range(ant_simulations):
        observation = env.reset()
        score_in_current_simulation = 0
        frozen_steps = 0
        previous_observation = [12345]
        for t in range(simulation_max_steps):
            if render:
                env.render()
            # print(observation)
            # action = env.action_space.sample()
            if use_observation_noise:
                observation = noizify_observations(observation)
            action = nerualNetwork.run(observation)
            if g > 3 :
                print(action)
            observation, reward, done, info = env.step(action)
            if (observation == previous_observation).all():
                frozen_steps+=1
                if frozen_steps> 5:
                    done = True
                    frozen_steps = 0

            else:
                frozen_steps=0
            previous_observation = observation
            score_in_current_simulation += reward
            if done:
                if render:
                    print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation ))
                total_score.append(score_in_current_simulation)
                break

    fitness = sum(total_score) / ant_simulations
    # fitness = min(total_score)
    individual.fitness = fitness
    # print("Average score for individual {}:  {}".format(individual.id,fitness))

    return fitness


def create_new_individual():
    # genome = {"weights": [1, random.random(), 1, random.random(), 0, 1, random.random(), 1],
    #           "biases": [0, random.random(), 0, 0, 2, random.random(), 0, 0, ]}
    id = next(global_individ_id_counter)
    genome = create_new_genome(id, layer_sizes)
    individual = Individ(id= id, genome= genome)
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
    genome = individual.genome
    for chromosome_key, chromosome in vars(genome).items():
        # new_chromosome = []
        # if isinstance(genome.chromosomes[chromosome], list):
        for key in chromosome:
            chromosome[key].mutate()


def crossover(parent1, parent2):
    assert parent1 != parent2
    # Setup

    if parent1.fitness < parent2.fitness:
        parent1, parent2 = deepcopy(parent2), deepcopy(parent1)

    p1_genome = parent1.genome
    p2_genome = parent1.genome
    id = next(global_individ_id_counter)
    child = Individ(id= id, genome= Genome())
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

        child.genome.__setattr__(chromosome_key, child_chromsome)
    return child


def noizify_observations(observations):
    noise_mean = 0
    noise = numpy.random.normal(loc=noise_mean, scale=observation_noise_std, size=ant_input_neruons)
    observations = observations + noise
    return observations


def write_generation_result_tofile(
        best_fitness : float,average_fitness : float, generation : int, delimiter: str = ";"
):
    filename = save_score_file
    with open(filename, 'a+') as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow([generation, round(best_fitness, 4), round(average_fitness, 4)])

def write_individal_to_file(individ : dict, generation : int, delimiter = ";"):
    filename = getIndividSaveFile(save_best_individual_base , generation)
    with open(filename, 'w') as f:
        json.dump(individ.__str__(), f)

def getIndividSaveFile( save_best_individual_base , generation):
    return "{}{}.json".format(save_best_individual_base , generation )


if __name__ == '__main__':
    # todo : config object?

    env = gym.make('BipedalWalker-v2')
    ## Configuration values
    # Genome config
    genome_value_min = -1
    genome_value_max = 1
    # ant_hidden_layers = 0
    ant_input_neruons = 24
    ant_output_neruons = 4
    layer_sizes = [ant_input_neruons, 8, ant_output_neruons]
    # Simulation run config
    simulation_max_steps = 500
    generations = 20
    pop_size = 50
    ant_simulations = 20
    observation_noise_std = 0.05
    use_observation_noise = False
    use_action_noise = False
    action_noise_rate = 0.01
    # GA params
    mutation_rate = 0.05
    mutation_power = 0.4
    crossover_mix_genes_rate = 0.1
    parent_selection_weighted_choises = False

    # Playback settings
    # save_best_each = 2
    playback_folder = "playback-18-1"
    save_best_individual_folder = playback_folder +"/bestIndividuals"
    save_best_individual_base = save_best_individual_folder+"/bestIndividualInGeneration_"
    save_score_file_folder = playback_folder+"/fitnesses"
    save_score_file= save_score_file_folder+ "/results.csv"
    if not path.exists(playback_folder):
        makedirs(playback_folder)
        makedirs(save_best_individual_folder)
        makedirs(save_score_file_folder)

    global_neruon_id_counter = count(start=ant_output_neruons)

    ## Init population
    population = initialize_populatin(pop_size)

    # geneB =  {'id': 2219, 'genome': Genome(layer_chromosome={0: LayerGene(level=0, innovationNumber=None, activation_function=None), 1: LayerGene(level=1, innovationNumber=None, activation_function=None), 2: LayerGene(level=2, innovationNumber=None, activation_function=None)}, neuron_chromosome={-1: NeuronGene(innovationNumber=-1, parent_layer=0), -2: NeuronGene(innovationNumber=-2, parent_layer=0), -3: NeuronGene(innovationNumber=-3, parent_layer=0), -4: NeuronGene(innovationNumber=-4, parent_layer=0), -5: NeuronGene(innovationNumber=-5, parent_layer=0), -6: NeuronGene(innovationNumber=-6, parent_layer=0), -7: NeuronGene(innovationNumber=-7, parent_layer=0), -8: NeuronGene(innovationNumber=-8, parent_layer=0), 108: NeuronGene(innovationNumber=108, parent_layer=1), 109: NeuronGene(innovationNumber=109, parent_layer=1), 110: NeuronGene(innovationNumber=110, parent_layer=1), 111: NeuronGene(innovationNumber=111, parent_layer=1), 0: NeuronGene(innovationNumber=0, parent_layer=2), 1: NeuronGene(innovationNumber=1, parent_layer=2), 2: NeuronGene(innovationNumber=2, parent_layer=2), 3: NeuronGene(innovationNumber=3, parent_layer=2)}, weight_chromosome={1248: WeightGene(value=0.9433276604105858, innovationNumber=1248, transmitting_neuron=-1, receiving_neuron=108), 1249: WeightGene(value=1.3309991359146838, innovationNumber=1249, transmitting_neuron=-1, receiving_neuron=109), 1250: WeightGene(value=0.8296905620249604, innovationNumber=1250, transmitting_neuron=-1, receiving_neuron=110), 1251: WeightGene(value=0.6173854220434225, innovationNumber=1251, transmitting_neuron=-1, receiving_neuron=111), 1252: WeightGene(value=1.2315522142001747, innovationNumber=1252, transmitting_neuron=-2, receiving_neuron=108), 1253: WeightGene(value=0.1717471046074385, innovationNumber=1253, transmitting_neuron=-2, receiving_neuron=109), 1254: WeightGene(value=-0.8730469043090361, innovationNumber=1254, transmitting_neuron=-2, receiving_neuron=110), 1255: WeightGene(value=0.14062688160851422, innovationNumber=1255, transmitting_neuron=-2, receiving_neuron=111), 1256: WeightGene(value=0.09762408450695048, innovationNumber=1256, transmitting_neuron=-3, receiving_neuron=108), 1257: WeightGene(value=0.19577176253156636, innovationNumber=1257, transmitting_neuron=-3, receiving_neuron=109), 1258: WeightGene(value=-0.15640976446709376, innovationNumber=1258, transmitting_neuron=-3, receiving_neuron=110), 1259: WeightGene(value=-0.0463296916910145, innovationNumber=1259, transmitting_neuron=-3, receiving_neuron=111), 1260: WeightGene(value=0.4322197991572294, innovationNumber=1260, transmitting_neuron=-4, receiving_neuron=108), 1261: WeightGene(value=-0.4405215085801435, innovationNumber=1261, transmitting_neuron=-4, receiving_neuron=109), 1262: WeightGene(value=-0.6708439704209194, innovationNumber=1262, transmitting_neuron=-4, receiving_neuron=110), 1263: WeightGene(value=1.2137582682183132, innovationNumber=1263, transmitting_neuron=-4, receiving_neuron=111), 1264: WeightGene(value=-1.2051739653304212, innovationNumber=1264, transmitting_neuron=-5, receiving_neuron=108), 1265: WeightGene(value=-0.38875891149916797, innovationNumber=1265, transmitting_neuron=-5, receiving_neuron=109), 1266: WeightGene(value=-0.006313907304450916, innovationNumber=1266, transmitting_neuron=-5, receiving_neuron=110), 1267: WeightGene(value=-0.5561866113517258, innovationNumber=1267, transmitting_neuron=-5, receiving_neuron=111), 1268: WeightGene(value=0.3031599686327767, innovationNumber=1268, transmitting_neuron=-6, receiving_neuron=108), 1269: WeightGene(value=-0.8095897889960313, innovationNumber=1269, transmitting_neuron=-6, receiving_neuron=109), 1270: WeightGene(value=0.1586314698285315, innovationNumber=1270, transmitting_neuron=-6, receiving_neuron=110), 1271: WeightGene(value=0.9918726184859533, innovationNumber=1271, transmitting_neuron=-6, receiving_neuron=111), 1272: WeightGene(value=-0.7355105369662676, innovationNumber=1272, transmitting_neuron=-7, receiving_neuron=108), 1273: WeightGene(value=-0.3899354078985431, innovationNumber=1273, transmitting_neuron=-7, receiving_neuron=109), 1274: WeightGene(value=-0.4394729613030534, innovationNumber=1274, transmitting_neuron=-7, receiving_neuron=110), 1275: WeightGene(value=0.22655646691765957, innovationNumber=1275, transmitting_neuron=-7, receiving_neuron=111), 1276: WeightGene(value=0.8268646338432031, innovationNumber=1276, transmitting_neuron=-8, receiving_neuron=108), 1277: WeightGene(value=-0.848737706344886, innovationNumber=1277, transmitting_neuron=-8, receiving_neuron=109), 1278: WeightGene(value=-0.9770976777158128, innovationNumber=1278, transmitting_neuron=-8, receiving_neuron=110), 1279: WeightGene(value=0.30049549592053476, innovationNumber=1279, transmitting_neuron=-8, receiving_neuron=111), 1280: WeightGene(value=0.7395496008902067, innovationNumber=1280, transmitting_neuron=108, receiving_neuron=0), 1281: WeightGene(value=-0.6690782608690221, innovationNumber=1281, transmitting_neuron=108, receiving_neuron=1), 1282: WeightGene(value=-0.08775073763817653, innovationNumber=1282, transmitting_neuron=108, receiving_neuron=2), 1283: WeightGene(value=0.14110829925703278, innovationNumber=1283, transmitting_neuron=108, receiving_neuron=3), 1284: WeightGene(value=0.6927498596772106, innovationNumber=1284, transmitting_neuron=109, receiving_neuron=0), 1285: WeightGene(value=0.17024718669516548, innovationNumber=1285, transmitting_neuron=109, receiving_neuron=1), 1286: WeightGene(value=0.7420756748955472, innovationNumber=1286, transmitting_neuron=109, receiving_neuron=2), 1287: WeightGene(value=0.3721029843758649, innovationNumber=1287, transmitting_neuron=109, receiving_neuron=3), 1288: WeightGene(value=-1.1755119055162369, innovationNumber=1288, transmitting_neuron=110, receiving_neuron=0), 1289: WeightGene(value=0.16905377466895763, innovationNumber=1289, transmitting_neuron=110, receiving_neuron=1), 1290: WeightGene(value=0.012035969476737685, innovationNumber=1290, transmitting_neuron=110, receiving_neuron=2), 1291: WeightGene(value=-0.4754311329467581, innovationNumber=1291, transmitting_neuron=110, receiving_neuron=3), 1292: WeightGene(value=1.1958463640363421, innovationNumber=1292, transmitting_neuron=111, receiving_neuron=0), 1293: WeightGene(value=0.043463871930397635, innovationNumber=1293, transmitting_neuron=111, receiving_neuron=1), 1294: WeightGene(value=0.026669355679097284, innovationNumber=1294, transmitting_neuron=111, receiving_neuron=2), 1295: WeightGene(value=-0.03835137731020505, innovationNumber=1295, transmitting_neuron=111, receiving_neuron=3)}, bias_chromosome={108: BiasGene(value=0.7668089188110419, parent_neuron=108, innovationNumber=None), 109: BiasGene(value=0.6232083939584674, parent_neuron=109, innovationNumber=None), 110: BiasGene(value=0.1525173479877075, parent_neuron=110, innovationNumber=None), 111: BiasGene(value=0.38175093124723825, parent_neuron=111, innovationNumber=None), 0: BiasGene(value=0.46697282789873307, parent_neuron=0, innovationNumber=None), 1: BiasGene(value=0.49660603710994705, parent_neuron=1, innovationNumber=None), 2: BiasGene(value=0.6718369588691699, parent_neuron=2, innovationNumber=None), 3: BiasGene(value=0.736153780701598, parent_neuron=3, innovationNumber=None)}), 'fitness': -8.393776440787112}
    # layer_sizes = [ant_input_neruons, 4, ant_output_neruons]

    for g in range(generations):
        # print("Generation {}".format(g))
        ## Evaualte
        for individual in population:
            evaluate(individual, ant_simulations= ant_simulations)

        after_evaluation = time.time()
        ## Summarize current generation
        best = max(population, key=lambda individual: individual.fitness)
        average_population_fitness = mean([individual.fitness for individual in population])
        print("Best Individual in generation {}: id:{}, fitness {}".format(g, best.id, best.fitness))
        print("Populatation average in generation {} was {}".format(g, average_population_fitness))

        print('------------------------------')
        write_generation_result_tofile(
            best_fitness=best.fitness,average_fitness= average_population_fitness,generation=g
        )
        write_individal_to_file(individ=best,generation=g)

        # Set up population for next generation
        if g != generations - 1:
            population.sort(key=lambda individual: individual.fitness, reverse=True)

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
                parent_fitnesses = [x.fitness for x in potential_parents]
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
    best = max(population, key=lambda individual: individual.fitness)
    print("best Individual got score {}, and looked like:   {}".format(best.fitness, best))
    average_population_fitness = mean([individual.fitness for individual in population])
    print("average population fitness:   {}".format(average_population_fitness))


    ## playback

    for g in range(generations):
        best_individual_file =getIndividSaveFile(save_best_individual_base , g)
        print("Best in generation {}".format(g))
        with open(best_individual_file, 'r') as f:
            dict_data = json.load(f)
            best_individual = eval(dict_data)

            evaluate(best_individual, ant_simulations=3, render= False)

    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv(save_score_file, sep=";",names=["generation", "best fitness", "avg_fitness"])
    fig, ax = plt.subplots()
    ax.plot(data["generation"], data["best fitness"])
    # data.plot(x="generation", y="best fitness")
    plt.show()
    fig.savefig(save_score_file_folder+"/test")
    env.close()
