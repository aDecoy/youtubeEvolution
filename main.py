import gym
import numpy as np
from random import random
from copy import deepcopy
from statistics import  median, mean

env = gym.make('CartPole-v0')


def perceptron(input_values):
    weights = np.array([1,1,1,1])
    biases = np.array([0,0,0,0])
    signal_strenght= input_values * weights + biases
    #no activation funciton to avoid extreame values
    output_signal = signal_strenght.sum()

    if output_signal > 0:
        return 1
    else:
        return 0

def mutate(genome):
    mutation_rate  = 0.1
    mutate_power = 0.1
    for chromosome in genome:
        new_genes = []
        for gene_value in genome[chromosome]:
            if random() < mutation_rate:
                # either increase or decrease the value
                if random()<0.5:
                    new_genes.append(gene_value+mutate_power)
                else:
                    new_genes.append(gene_value-mutate_power)
            else:
                # No change in the value
                new_genes.append(gene_value)
        genome[chromosome] = new_genes
    return genome


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        genome = {"weights":[], "bias":[]}
        for _ in range(4):
            genome["weights"].append(random())
        for _ in range(4):
            genome["bias"].append(random())
        population.append(genome)
    return population

def evaluate(genome):
    observation = env.reset()
    for t in range(200):
        # env.render()
        # print(observation)
        action = perceptron(np.array(observation))
        observation, reward, done, info = env.step(action)
        if done:
            break
    # fitness is how many frames it stayed alive
    # print("Episode finished after {} timesteps".format(t + 1))
    return t

def run_evolution():
    env = gym.make('CartPole-v0')
    generations = 10
    pop_size = 20
    survival_rate = 0.5
    population = initialize_population(pop_size=pop_size)
    for generation in range(generations):
        all_fintesses = []
        for genome in population:
            all_fintesses.append(evaluate(genome))

        print("Generation : {}".format(generation))
        print("Population size: {}".format(len(population)))
        print("Population average fitness {}, best individual: {} ".format(mean(all_fintesses),max(all_fintesses)))


        # Selection
        # median_fintess = median(all_fintesses)
        fitness_treshold = sorted(all_fintesses,reverse=True)[int(pop_size*survival_rate)]
        survivors = []
        for i in range(len(population)):
            if all_fintesses[i]>= fitness_treshold and len(survivors)<pop_size*survival_rate:
                survivors.append(population[i])

        # Mutation
        new_genomes = []
        for genome in survivors:
            child = deepcopy(genome)
            mutate(child)
            new_genomes.append(child)

        population = survivors + new_genomes

    env.close()


if __name__ == '__main__':
    run_evolution()