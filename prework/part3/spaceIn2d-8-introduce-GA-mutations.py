import gym
import numpy
import torch
import random
from copy import deepcopy
from statistics import median, mean, pvariance,variance, stdev , pstdev
torch.no_grad()

ant_input, ant_output = 8, 1

genome_value_min, genome_value_max = -1,1

class Perceptron():
    def __init__(self, genome=None):

        if genome is None:
            genome = {"weights": [1, 1, 1, 1, 1, 1, 1, 1], "biases": [0, 0, 0, 0, 0, 0, 0, 0]}

        weights = genome["weights"]
        biases = genome["weights"]

        weights = numpy.array(weights )
        biases = numpy.array(biases )
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
        else :
            return 3

def initialize_population(pop_size):
    population = []
    nodes = 8
    for _ in range(pop_size):
        genome = {"weights":[], "bias":[]}
        for _ in range(8):
            genome["weights"].append(random.uniform(genome_value_min,genome_value_max))
        for _ in range(8):
            genome["bias"].append(random.uniform(genome_value_min,genome_value_max))
        population.append({"genome":genome})
    return population

def replenish_population_with_random_genomes(pop_size, population ):
    nodes = 8
    individuals_killed = pop_size - len(population)

    for _ in range(individuals_killed):
        genome = {"weights":[], "bias":[]}
        for _ in range(8):
            genome["weights"].append(random.uniform(genome_value_min,genome_value_max))
        for _ in range(8):
            genome["bias"].append(random.uniform(genome_value_min,genome_value_max))
        population.append({"genome":genome})
    return population

def mutate(individual):
    mutation_rate  = 0.3
    mutate_power = 0.1
    genome = individual['genome']
    for chromosome in genome:
        new_genes = []
        for old_gene_value in genome[chromosome]:
            if random.random() < mutation_rate:
                # Either increase or decrease the value
                if random.random()<0.5:
                    new_genes.append(old_gene_value+mutate_power)
                else:
                    new_genes.append(old_gene_value-mutate_power)
            else:
                # No change in the value
                new_genes.append(old_gene_value)
        genome[chromosome] = new_genes

    individual["genome"] = genome
    return individual

def evaluate(individual_genome):
    total_score = 0
    perceptron = Perceptron()
    simulation_scores =[]
    for i_episode in range(ant_simulations):
        observation = env.reset()
        reward_in_current_simulation = 0
        for t in range(simulation_max_timesteps):
            # env.render()
            # print(observation)
            action = perceptron.run(observation)
            observation, reward, done, info = env.step(action)
            reward_in_current_simulation += reward
            if done:
                # print("Episode finished after {} timesteps. Reward: {}".format(t + 1, reward_in_current_simulation))
                simulation_scores.append(reward_in_current_simulation)
                break
    individual_genome["fitness"] = sum(simulation_scores) / ant_simulations
    individual_genome["stdev"] = pstdev(simulation_scores)

https://hal.archives-ouvertes.fr/hal-02444350/document
if __name__ == '__main__':
    # Configuration
    env = gym.make('LunarLander-v2')
    ant_simulations = 40
    simulation_max_timesteps = 250
    pop_size = 100
    generations = 10

    ## Init population
    population = initialize_population(pop_size)

    ### Running stuff
    for _ in range(generations):

        for individual in population:
            evaluate(individual)
            # print("Individual {} got an average score {}".format( population.index(individual),individual["fitness"]))
            print("individual had stdev : {}".format(individual["stdev"]))

        population.sort(key=lambda x: x['fitness'],reverse=True)
        # print([x['fitness'] for x in population])

        # Generation Summary
        best = max(population, key=lambda x: x['fitness'])
        print("Best fitness achived was : {}".format(best["fitness"]))
        print("Best individual was : {}".format(best))
        print("Average fintess in generation was {}".format(mean(x['fitness'] for x in population)))

        # Keep 5 best, kill the rest!
        parents_percentage = 0.1
        ant_parents = round(pop_size*parents_percentage)
        parents = population[0:ant_parents]
        children = []
        ant_random_new = 20
        ant_children = pop_size - len(parents) - ant_random_new

        for i in range(ant_children):
            child = deepcopy(parents[random.randrange(0,len(parents))])
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

