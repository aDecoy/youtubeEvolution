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

#Best individual was : {'id': 1826, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 0.0}

# https://hal.archives-ouvertes.fr/hal-02444350/document
# Best fitness achived was : -20.533515181022352
# Best individual was : {'id': 1760, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.12666600979486442, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': -20.533515181022352}
# --------------------------------------------------
# Best fitness achived was : 0.0
# Best individual was : {'id': 1826, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 0.0}
# --------------------------------------------------
# Best fitness achived was : 0.0
# Best individual was : {'id': 2139, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.12666600979486442, 0.9589173719427644, 0.0717536726600656, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 0.0}
# --------------------------------------------------
# Best fitness achived was : 0.0
# Best individual was : {'id': 2392, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.4549523056883864, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 0.0}
# --------------------------------------------------
# Best fitness achived was : 2.3555095271481905
# Best individual was : {'id': 2687, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 2.3555095271481905}
# --------------------------------------------------
# Best fitness achived was : 1.343593063809628
# Best individual was : {'id': 3082, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 1.343593063809628}
# --------------------------------------------------
# Best fitness achived was : 11.547230542273159
# Best individual was : {'id': 3391, 'genome': {'weights': [0.0762351887199717, -0.07954399145330426, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 11.547230542273159}
# --------------------------------------------------
# Best fitness achived was : 3.9214848702943383
# Best individual was : {'id': 3723, 'genome': {'weights': [0.0762351887199717, -0.07954399145330426, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 3.9214848702943383}
# --------------------------------------------------
# Best fitness achived was : 18.209972255304052
# Best individual was : {'id': 4116, 'genome': {'weights': [0.0762351887199717, -0.07954399145330426, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 18.209972255304052}
# --------------------------------------------------
# Best fitness achived was : 2.465901138562148
# Best individual was : {'id': 4342, 'genome': {'weights': [0.0762351887199717, -0.07954399145330426, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, 0.5908965319097161], 'biases': [-0.5500036853587174, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 2.465901138562148}
# --------------------------------------------------
# Best fitness achived was : 15.841208863405885
# Best individual was : {'id': 4624, 'genome': {'weights': [0.0762351887199717, -0.07954399145330426, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, -0.42222519370978917], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 15.841208863405885}
# --------------------------------------------------
# a = {'id': 4624, 'genome': {'weights': [0.0762351887199717, -0.07954399145330426, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.7172755926846484, -0.42222519370978917], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}}
# a = {'id': 2392, 'genome': {'weights': [0.0762351887199717, -0.02640922493914033, -0.9923665515655753, 0.37445495060686707, 0.7249657205728088, 0.9589173719427644, -0.4549523056883864, 0.5908965319097161], 'biases': [-0.6877587760515107, 0.7998630364793076, 0.3349204632814373, 0.8614082050943472, 0.3310104313479525, 0.31334060900556127, 0.8415308789975724, -0.5521768506083535]}, 'fitness': 0.0}
if __name__ == '__main__':
    # Configuration
    env = gym.make('LunarLander-v2')
    ant_simulations = 3
    simulation_max_timesteps = 250
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
            children = []
            random_new_percentage = 0.01
            ant_random_new = ceil(pop_size * random_new_percentage)
            ant_children = pop_size - len(parents) - ant_random_new

            for i in range(ant_children):
                child = deepcopy(parents[random.randrange(0, len(parents))])
                child["id"]=next(global_individ_id_counter)
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
