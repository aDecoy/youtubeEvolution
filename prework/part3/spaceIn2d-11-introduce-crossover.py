import gym
import numpy
import torch
import random
from copy import deepcopy
from statistics import median, mean, pvariance, variance, stdev, pstdev
from math import ceil
from itertools import count

torch.no_grad()

ant_input, ant_output = 8, 1

genome_value_min, genome_value_max = -1, 1

global_individ_id_counter = count()

class Perceptron():
    def __init__(self, genome=None):

        if genome is None:
            genome = {"weights": [1, 1, 1, 1, 1, 1, 1, 1], "biases": [0, 0, 0, 0, 0, 0, 0, 0]}

        weights = genome["weights"]
        biases = genome["biases"]

        weights = numpy.array(weights)
        biases = numpy.array(biases)
        self.weights = weights
        self.biases = biases

    def run(self, input_observations):
        signal_strength = input_observations * self.weights + self.biases
        output_signal = signal_strength.sum()

        if output_signal < 1:
            return 0
        elif output_signal < 2:
            return 1
        elif output_signal < 3:
            return 2
        else:
            return 3


def initialize_population(pop_size):
    population = []
    nodes = 8
    for _ in range(pop_size):
        genome = {"weights": [], "biases": []}
        for _ in range(8):
            genome["weights"].append(random.uniform(genome_value_min, genome_value_max))
        for _ in range(8):
            genome["biases"].append(random.uniform(genome_value_min, genome_value_max))
        population.append({"id": next(global_individ_id_counter),"genome": genome})
    return population


def replenish_population_with_random_genomes(pop_size, population):
    nodes = 8
    individuals_killed = pop_size - len(population)

    for _ in range(individuals_killed):
        genome = {"weights": [], "biases": []}
        for _ in range(8):
            genome["weights"].append(random.uniform(genome_value_min, genome_value_max))
        for _ in range(8):
            genome["biases"].append(random.uniform(genome_value_min, genome_value_max))
        population.append({"id": next(global_individ_id_counter),"genome": genome})
    return population


def mutate(individual):
    mutation_rate = 0.15
    mutate_power = 0.4
    genome = individual['genome']
    for chromosome in genome:
        new_genes = []
        for old_gene_value in genome[chromosome]:
            if random.random() < mutation_rate:
                # Either increase or decrease the value . Alternativly just random
                # new_genes.append(random.uniform(genome_value_min, genome_value_max))
                new_genes.append(old_gene_value+ random.uniform( -mutate_power,mutate_power))

            else:
                # No change in the value
                new_genes.append(old_gene_value)
        genome[chromosome] = new_genes

    individual["genome"] = genome
    return individual

def crossover(parent1,parent2):
    p1_genome = parent1["genome"]
    p2_genome = parent2["genome"]
    child = {"id": next(global_individ_id_counter), "genome":{}}
    for chromosome_key, parent1_chromosome in p1_genome.items():
        parent2_chromosome = p2_genome[chromosome_key]
        child_chromosome = deepcopy(parent1_chromosome)
        for i in range(len(child_chromosome)):
            crossover_event = random.random()
            if crossover_event < 0.6:
                child_chromosome[i] = (parent1_chromosome[i] + parent2_chromosome[i]) / 2
            elif crossover_event < 0.8 :
                child_chromosome[i] = parent2_chromosome[i]
            # else keep parent 1 value
        child["genome"][chromosome_key] = child_chromosome

    return child




def evaluate(individual_genome):
    total_score = 0
    perceptron = Perceptron(individual["genome"])
    # perceptron = Perceptron(a["genome"])
    simulation_scores = []
    for i_episode in range(ant_simulations):
        observation = env.reset()
        reward_in_current_simulation = 0
        for t in range(simulation_max_timesteps):
            # env.render()
            # print(observation)
            action = perceptron.run(observation)
            observation, reward, done, info = env.step(action)
            reward_in_current_simulation += reward
            # print(reward)
            if done or t == simulation_max_timesteps-1:
                # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, reward_in_current_simulation))
                simulation_scores.append(reward_in_current_simulation)
                break
    individual_genome["fitness"] = sum(simulation_scores) / ant_simulations
    # individual_genome["stdev"] = pstdev(simulation_scores)

if __name__ == '__main__':
    # Configuration
    env = gym.make('LunarLander-v2')
    ant_simulations = 3
    simulation_max_timesteps = 450
    pop_size = 200
    generations = 200

    ## Init population
    population = initialize_population(pop_size)

    ### Running stuff
    generation_best = []
    for g in range(generations):

        for individual in population:
            evaluate(individual)
            # print("Individual {} got an average score {}".format( population.index(individual),individual["fitness"]))
            # print("individual had stdev : {}".format(individual["stdev"]))

        population.sort(key=lambda x: x['fitness'], reverse=True)
        # print([x['fitness'] for x in population])

        # Generation Summary
        best = max(population, key=lambda x: x['fitness'])
        generation_best.append(best["fitness"])
        print("Best fitness achived was : {}".format(best["fitness"]))
        print("Best individual in genertation {} was : {}".format(g,best))
        # print("Average fintess in generation was {}".format(mean(x['fitness'] for x in population)))
        print('--------------------------------------------------')
        if g != generations - 1:
            # Keep 5 best, kill the rest!
            parents_percentage = 0.1
            ant_parents = ceil(pop_size * parents_percentage)
            # ant_parents = 3
            parents = population[0:ant_parents]
            # parent_fitnesses = [x['fitness'] for x in parents]
            # parents_fitness_range = abs(parent_fitnesses[0] - parent_fitnesses[-1])
            # normalized_probabilities = [round(abs(fitness-parent_fitnesses[-1]) / parents_fitness_range, 2) for fitness in parent_fitnesses]
            # sumOfP = sum(normalized_probabilities)
            # reproduction_probability = [round(p / sumOfP, 2) for p in normalized_probabilities]
            # print(reproduction_probability)
            children = []
            random_new_percentage = 0.01
            ant_random_new = ceil(pop_size * random_new_percentage)
            ant_children = pop_size - len(parents) - ant_random_new

            for i in range(ant_children):
                # child = deepcopy(parents[random.randrange(0, len(parents))])
                # parent1, parent2 = random.choices(parents, weights= reproduction_probability,k=2)
                parent1, parent2 = random.choices(parents,k=2)

                # child = deepcopy(random.choices(parents, weights= reproduction_probability))[0]
                # child["id"]=next(global_individ_id_counter)
                child  = crossover(parent1,parent2)
                children.append(mutate(child))

            population = parents + children
            population = replenish_population_with_random_genomes(pop_size, population)
            assert len(population) == pop_size
        # only gives random. what about using what we know works

    # Summary
    best = max(population, key=lambda x: x['fitness'])
    print("Best fitness achived was : {}".format(best["fitness"]))
    print("Best individual was : {}".format(best))
    env.close()
