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
global_weight_chromsome_id_counter = count()
global_bias_chromsome_id_counter = count()


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

    def mutate(self):
        self.value += random.uniform(-mutation_power, +mutation_power)


@dataclass
class WeightGene:
    innovationNumber: int
    value: float
    transmitting_neuron: int
    receiving_neuron: int

    def mutate(self):
        self.value += random.uniform(-mutation_power, +mutation_power)


@dataclass
class NeruonGene:
    innovationNumber: int
    parent_layer: int

    def mutate(self):
        pass


@dataclass
class LayerGene:
    innovationNumber: int

    # later add activation function
    def mutate(self):
        pass


def create_new_layer_chromosome(ant_layers):
    return {i: LayerGene(innovationNumber=i) for i in range(ant_layers)}


def create_new_neruon_chromosome(layer_sizes):
    neurons = {}
    input_innovation_number = -1

    # Input neurons
    for i in range(layer_sizes[0]):
        neurons[input_innovation_number] = NeruonGene(innovationNumber=input_innovation_number, parent_layer=0)
        input_innovation_number -= 1

    # Hidden neurons
    for layer, ant_neurons in enumerate(layer_sizes[1:-1], start=1):
        for i in range(ant_neurons):
            id = next(global_neuron_chromsome_id_counter)
            neurons[id] = NeruonGene(innovationNumber=id, parent_layer=layer)
    # Output neruon
    output_innovation_number = 0
    for i in range(layer_sizes[-1]):
        id = output_innovation_number
        output_innovation_number += 1
        neurons[id] = NeruonGene(innovationNumber=id, parent_layer=len(layer_sizes) - 1)

    return neurons


def create_new_weight_chromosome(neuron_choromosome: dict, ant_layers: int):
    weights = {}
    # input neurons
    transmitting_neurons = {key: gene for key, gene in neuron_choromosome.items() if gene.parent_layer == 0}

    # Loop over the rest of the layers, finding values for all weights
    for receiving_layer in range(1, ant_layers):
        receiving_neurons = {key: gene for key, gene in neuron_choromosome.items() if gene.parent_layer == receiving_layer}

        # for each gene
        for transmitting_neuron in transmitting_neurons:
            for receiving_neuron in receiving_neurons:
                id = next(global_weight_chromsome_id_counter)
                weights[id] = WeightGene(innovationNumber=id, transmitting_neuron=transmitting_neuron, receiving_neuron=receiving_neuron,
                                         value=random.uniform(genome_value_min, genome_value_max))

        transmitting_neurons = receiving_neurons

    return weights


def create_new_bias_chromosome(neuron_chromosome):
    biases = {}
    for neuron_id in neuron_chromosome:
        if neuron_id >= 0:
            biases[neuron_id] = BiasGene(innovationNumber=neuron_id, parent_neuron=neuron_id,
                                         value=random.uniform(genome_value_min, genome_value_max))
    return biases


def create_new_genome(layer_sizes=[8, 4, 1]):
    layers = create_new_layer_chromosome(len(layer_sizes))
    neurons = create_new_neruon_chromosome(layer_sizes)
    weights = create_new_weight_chromosome(neurons, ant_layers=len(layer_sizes))
    biases = create_new_bias_chromosome(neuron_chromosome=neurons)

    return Genome(layer_chromosome=layers,
                  neuron_chromosme=neurons,
                  weight_chromosme=weights,
                  bias_chromosme=biases)


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
    def __init__(self, phenotype: Phenotype):
        # 8 input, 4 hidden , 1 output
        # if genome is None:

        self.weights = phenotype.weights
        self.biases = phenotype.biases

    def noise_output(self):
        return random.choice(range(3))

    def run(self, input_values):

        if use_action_noise:
            if random.random() < action_noise_rate:
                return self.noise_output()

        input_values = numpy.array(input_values).reshape(-1)

        signal_strength_1_to_2 = numpy.matmul(self.weights["0-1"], input_values) + self.biases[1]
        signal_strength_2_to_3 = numpy.matmul(self.weights["1-2"], signal_strength_1_to_2) + self.biases[2]
        output_signal = signal_strength_2_to_3.sum()
        # print(output_signal)
        if output_signal < 0.25:
            return 1
        elif output_signal < 0.50:
            return 0
        elif output_signal < 0.75:
            return 3
        else:  # 0.75 to 1
            return 2


def evaluate(individual):
    total_score = []
    phenotype = develop_genome_to_phenotype(individual["genome"])
    neuralNetwork = NeuralNetworkModel(phenotype)
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
            action = neuralNetwork.run(observation)
            observation, reward, done, info = env.step(action)
            score_in_current_simulation += reward
            if done:
                # print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation ))
                total_score.append(score_in_current_simulation)
                break

    fitness = sum(total_score) / ant_simulations
    individual["fitness"] = fitness
    # print("Average score for individual {}".format(fitness))

    return fitness


def create_new_individual():
    genome = create_new_genome()
    # genome = {"weights": [], "biases": []}
    # for chromosome in genome:
    #     for i in range(8):
    #         genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    #     # genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    individual = {"id": next(global_individ_id_counter), "genome": genome}
    return individual


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
        for gene_key in chromosome:
            if random.random() < mutation_rate:
                chromosome[gene_key].mutate()


def crossover(parent1: dict, parent2: dict):
    assert parent1 != parent2

    if parent1["fitness"] < parent2["fitness"]:
        parent1, parent2 = deepcopy(parent2), deepcopy(parent1)

    # Setup
    p1_genome = parent1["genome"]
    p2_genome = parent1["genome"]
    child = {"id": next(global_individ_id_counter), "genome": Genome()}
    # Mix the genes

    chromosomes_that_can_average_values = ["weight_chromosome", "bias_chromosome"]

    # crossover neruons
    for chromosome_key, parent1_chromosome in vars(p1_genome).items():
        parent2_chromosome = p2_genome.__getattribute__(chromosome_key)

        child_chromsome = deepcopy(parent1_chromosome)
        for gene_id in child_chromsome:
            if gene_id in parent2_chromosome:
                # cross over
                crossover_event = random.random()
                if crossover_event < crossover_mix_genes_rate and chromosome_key in chromosomes_that_can_average_values:
                    child_chromsome[gene_id] = (parent1_chromosome[gene_id].value + parent2_chromosome[gene_id].value) * 0.5
                elif crossover_event < crossover_mix_genes_rate + (1 - crossover_mix_genes_rate) / 2:
                    # keep gene from parent 2
                    child_chromsome[gene_id] = parent2_chromosome[gene_id]
            else:
                # keep gene from parent 1
                pass

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
    # Simulation run config
    simulation_max_steps = 350
    generations = 50
    pop_size = 100
    # ant_simulations = 15
    ant_simulations = 30
    observation_noise_std = 0.05
    use_observation_noise = False
    use_action_noise = False
    action_noise_rate = 0.1
    # GA params
    mutation_rate = 0.05
    mutation_power = 0.4
    crossover_mix_genes_rate = 0.1
    parent_selection_weighted_choises = False

    ant_input_neurons = 8
    ant_output_neurons = 1
    layer_sizes = [ant_input_neurons, 4, ant_output_neurons]
    global_neuron_chromsome_id_counter = count(start=layer_sizes[ant_output_neurons])

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
            parent_rate = 0.1
            ant_parents = max(round(parent_rate * pop_size), 2)
            potential_parents = population[0:ant_parents]

            # parent_fitnesses = [ x["fitness"] for x in potential_parents]
            # parent_fitnesses_range = abs ( parent_fitnesses[0] - parent_fitnesses[-1])
            # normalize_fitness_performance = [round(abs(fitness - parent_fitnesses[-1]))/ parent_fitnesses_range for fitness in parent_fitnesses]
            # sumOfP = sum(normalize_fitness_performance)
            # reproduction_probability = [p / sumOfP for p in normalize_fitness_performance]

            ant_children = len(population) - ant_survivors - ant_random_new
            children = []
            for i in range(ant_children):
                # parent = potential_parents[random.randrange(0, len(potential_parents))]
                # Parent selection
                # if parent_selection_weighted_choises:
                #     parent1,parent2 = numpy.random.choice(potential_parents,size=2, p= reproduction_probability,replace=False)
                # else:
                parent1, parent2 = random.sample(potential_parents, k=2)

                # parent1, parent2 = random.choice(potential_parents, 2, weights= reproduction_probability)  # todo weighted selection

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
