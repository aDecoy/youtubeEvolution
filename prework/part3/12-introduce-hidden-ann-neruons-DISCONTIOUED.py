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
    layer_chromosme : dict = None
    neruon_chromosme : dict = None
    weight_chromosome: dict = None
    bias_chromosome: dict = None

@dataclass
class Gene:
    innovationNumber: int = None

@dataclass
class BiasGene:
    value : float
    innovationNumber: int
    parent_neruon : int

@dataclass
class WeightGene:
    value : float
    innovationNumber: int
    transmitting_neruon : int
    receiving_neruon : int

@dataclass
class NeruonGene:
    innovationNumber: int
    parent_layer : int

@dataclass
class LayerGene:
    innovationNumber: int


@dataclass
class Phenotype:
    weights :dict
    biases : dict

class NeuralNetworkModel:

    # def __init__(self, phenotype : Phenotype):
    def __init__(self, genome={"weights": [2, 0.9, 1.2, 1.1, 1, 1, 1, 1], "biases": [0, 0, 0, 0, 0, 0, 0, 0, ]}):
        self.weights = numpy.array(genome["weights"])
        self.biases = numpy.array(genome["biases"])

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
        #
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

    if genome is None:

        genome = Genome(id=None,
                                layer_chromosome = {0:LayerGene(level=0 ),1:LayerGene(level=1 ),2:LayerGene(level=2 )}
                                neuron_chromosome = {0: NeuronGene(innovationNumber=0, parent_layer=0),
                                                     1: NeuronGene(innovationNumber=1, parent_layer=0),
                                                     2: NeuronGene(innovationNumber=2, parent_layer=0),
                                                     3: NeuronGene(innovationNumber=3, parent_layer=0),
                                                     4: NeuronGene(innovationNumber=4, parent_layer=0),
                                                     5: NeuronGene(innovationNumber=5, parent_layer=0),
                                                     6: NeuronGene(innovationNumber=6, parent_layer=0),
                                                     7: NeuronGene(innovationNumber=7, parent_layer=0),

                                                     8: NeuronGene(innovationNumber=8, parent_layer=1),
                                                     9: NeuronGene(innovationNumber=9, parent_layer=1),
                                                     10: NeuronGene(innovationNumber=10, parent_layer=1),
                                                     11: NeuronGene(innovationNumber=11, parent_layer=1),

                                                     12: NeuronGene(innovationNumber=12, parent_layer=2)
                                                     },
                                bias_chromosome = {0: BiasGene(parent_neuron=0, value=0),
                                                     1: BiasGene(parent_neuron=1, value=0),
                                                     2: BiasGene(parent_neuron=2, value=0),
                                                     3: BiasGene(parent_neuron=3, value=0),
                                                     4: BiasGene(parent_neuron=4, value=0),
                                                     5: BiasGene(parent_neuron=5, value=0),
                                                     6: BiasGene(parent_neuron=6, value=0),
                                                     7: BiasGene(parent_neuron=7, value=0),

                                                     8: BiasGene(parent_neuron=8, value=1),
                                                     9: BiasGene(parent_neuron=9, value=1),
                                                     10: BiasGene(parent_neuron=10, value=1),
                                                     11: BiasGene(parent_neuron=11, value=1),

                                                     12: BiasGene(parent_neuron=12, value=2)
                                                     },
                                chromosomes={Chromosome(name="weights",
                                                        genes={
                                                            "layer_0": [], # here the activation function will been
                                                            "layer_1": [],
                                                            "layer_2" : [],
                                                            "0-1": [
                                                                [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
                                                                [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
                                                                [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)],
                                                                [Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1), Gene(1)]],
                                                            "1-2": [[Gene(1), Gene(1), Gene(1), Gene(1)],
                                                                    [Gene(1), Gene(1), Gene(1), Gene(1)],
                                                                    [Gene(1), Gene(1), Gene(1), Gene(1)],
                                                                    [Gene(1), Gene(1), Gene(1), Gene(1)]]
                                                        }),
                                             Chromosome(name="biases",
                                                        genes={"1": [Gene(0), Gene(0), Gene(0), Gene(0)], "2": [Gene(0)]})
                                             }
                                )

    def noise_output(self):
        return random.choice(range(3))

    def run(self, input_values):

        if use_action_noise:
            if random.random() < action_noise_rate:
                return self.noise_output()

        signal_strength = input_values * self.weights + self.biases
        output_signal = signal_strength.sum()

        if output_signal < 0.25:
            return 0
        elif output_signal < 0.50:
            return 1
        elif output_signal < 0.75:
            return 2
        else:  # 0.75 to 1
            return 3


def evaluate(individual):
    total_score = []
    perceptron = NeuralNetworkModel(individual["genome"])
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
            action = perceptron.run(observation)
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
    # genome = {"weights": [1, random.random(), 1, random.random(), 0, 1, random.random(), 1],
    #           "biases": [0, random.random(), 0, 0, 2, random.random(), 0, 0, ]}
    genome = {"weights": [], "biases": []}
    for chromosome in genome:
        for i in range(8):
            genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
        # genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
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
    for chromosome in genome:
        new_chromosome = []
        for old_gene_value in genome[chromosome]:
            if random.random() < mutation_rate:
                # Mutate
                new_chromosome.append(old_gene_value + random.uniform(-mutation_power, +mutation_power))
            else:
                new_chromosome.append(old_gene_value)

        genome[chromosome] = new_chromosome


def crossover(parent1, parent2):
    assert parent1 != parent2
    # Setup
    p1_genome = parent1["genome"]
    p2_genome = parent1["genome"]
    child = {"id": next(global_individ_id_counter), "genome": {}}
    # Mix the genes

    for chromosome_key, parent1_chromsome in p1_genome.items():
        parent2_chromsome = p2_genome[chromosome_key]
        child_chromsome = deepcopy(parent1_chromsome)
        # Cross over chromoome
        for i in range(len(child_chromsome)):
            crossover_event = random.random()
            if crossover_event < crossover_mix_genes_rate:
                child_chromsome[i] = (parent1_chromsome[i] + parent2_chromsome[i]) * 0.5
            elif crossover_event < crossover_mix_genes_rate + (1 - crossover_mix_genes_rate) / 2:
                child_chromsome[i] = parent2_chromsome[i]
            ##else keep parent 1

        child["genome"][chromosome_key] = child_chromsome
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
