import gym
import numpy
import torch
import random
torch.no_grad()

ant_input, ant_output = 8, 1


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
            genome["weights"].append(random.random())
        for _ in range(8):
            genome["bias"].append(random.random())
        population.append(genome)
    return population

def replenish_population(pop_size, population = []):
    nodes = 8
    individuals_killed = pop_size - len(population)

    for _ in range(individuals_killed):
        genome = {"weights":[], "bias":[]}
        for _ in range(8):
            genome["weights"].append(random.random())
        for _ in range(8):
            genome["bias"].append(random.random())
        population.append(genome)
    return population

def evaluate(individual_genome):
    total_score = 0
    perceptron = Perceptron()
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
                total_score += reward_in_current_simulation
                break
    individual_genome["fitness"] = total_score / ant_simulations

if __name__ == '__main__':
    # Configuration
    env = gym.make('LunarLander-v2')
    ant_simulations = 20
    simulation_max_timesteps = 250
    pop_size = 20
    generations = 5

    ## Init population
    population = initialize_population(pop_size)

    ### Running stuff
    for _ in range(generations):

        for individual in population:
            evaluate(individual)
            print("Individual {} got an average score {}".format( population.index(individual),individual["fitness"]))

        population.sort(key=lambda x: x['fitness'],reverse=True)

        # Keep 5 best, kill the rest!
        new_population = population[0:5]
        population = replenish_population(pop_size, population)
        assert len(population) == pop_size
        # only gives random. what about using what we know works

    # Summary
    best = max(population, key=lambda x: x['fitness'])
    print("Best fitness achived was : {}".format(best["fitness"]))
    print("Best individual was : {}".format(best))
    env.close()

