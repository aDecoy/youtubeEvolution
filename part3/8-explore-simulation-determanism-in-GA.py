import gym
import numpy
import random
from statistics import mean, stdev, pstdev
from itertools import count
from copy import deepcopy
from collections import Counter

global_individ_id_counter = count()


class NeuralNetworkModel:
    def __init__(self, genome={"weights": [2, 0.9, 1.2, 1.1, 1, 1, 1, 1], "biases": [0, 0, 0, 0, 0, 0, 0, 0, ]}):
        self.weights = numpy.array(genome["weights"])
        self.biases = numpy.array(genome["biases"])

    def run(self, input_values):

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


def mutate(individual):
    mutation_rate = 0.15
    mutation_power = 0.4
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


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    ## Configuration values
    simulation_max_steps = 350
    genome_value_min = -1
    genome_value_max = 1
    generations = 100
    pop_size = 50
    ant_simulations = 1
    crossover_mix_genes_rate = 0.1

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
            parent_rate = 0.05
            ant_parents = max(round(parent_rate * pop_size), 2)
            potential_parents = population[0:ant_parents]

            ant_children = len(population) - ant_survivors - ant_random_new
            children = []
            for i in range(ant_children):
                # parent = potential_parents[random.randrange(0, len(potential_parents))]
                # Parent selection
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
