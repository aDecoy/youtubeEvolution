import gym
import numpy
import random
from statistics import mean, stdev, pstdev
from itertools import count
from copy import deepcopy

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
    for i_episode in range(ant_simulations):
        observation = env.reset()
        score_in_current_simulation = 0
        for t in range(simulation_max_steps):
            # env.render()
            # print(observation)
            action = env.action_space.sample()
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
    genome = {"weights": [], "biases": []}
    for chromosome in genome:
        for i in range(8):
            genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    individual = {"id": next(global_individ_id_counter), "genome": genome}
    individual["ancestor"] = individual["id"]
    return individual


def initialize_populatin(pop_size):
    population = []
    for _ in range(pop_size):
        population.append(create_new_individual())
    return population


def replenish_population(pop_size, population):
    individuals_killed = pop_size - len(population)
    for _ in range(individuals_killed):
        genome = {"weights": [], "biases": []}
        for chromosome in genome:
            for i in range(8):
                genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
        population.append({"id": next(global_individ_id_counter), "genome": genome})


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

    genome["id"] = next(global_individ_id_counter)


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    ## Configuration values
    ant_simulations = 20
    simulation_max_steps = 350
    genome_value_min = -1
    genome_value_max = 1
    generations = 50
    pop_size = 50

    ## Init population
    population = initialize_populatin(pop_size)

    ### Run stuff (evolve the population to find the best individual)

    for g in range(generations):
        print("Generation {}".format(g))
        ## Evaualte
        for individual in population:
            evaluate(individual)

        ## Summarize current generation
        best = max(population, key=lambda individual: individual["fitness"])
        average_population_fitness = mean([individual["fitness"] for individual in population])
        print("Best Individual in generation {}: id:{}, fitness {}".format(g, best["id"], best["fitness"]))
        print("Current populatation average in generation {} was {}".format(g, average_population_fitness))

        # Set up population for next generation
        if g != generations - 1:
            # Keep the 5 best. Kill the rest!
            population.sort(key=lambda individual: individual["fitness"], reverse=True)

            survival_rate = 0.1
            ant_survivors = round(len(population) * survival_rate)

            survivers = population[0:ant_survivors]

            ant_children = len(population) - ant_survivors
            potential_parents = survivers
            children = []
            for i in range(ant_children):
                parent = potential_parents[random.randrange(0, len(potential_parents))]
                child = deepcopy(parent)
                mutate(child)
                children.append(mutate(child))

            # replenish_population(pop_size,survivers)
            population = survivers + children
            assert len(population) == pop_size

    ### Summary
    best = max(population, key=lambda individual: individual["fitness"])
    print("best Individual got score {}, and looked like:   {}".format(best["fitness"], best))
    average_population_fitness = mean([individual["fitness"] for individual in population])
    print("average population fitness:   {}".format(average_population_fitness))

    env.close()
