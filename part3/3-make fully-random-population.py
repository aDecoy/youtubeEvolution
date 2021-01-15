import gym
import numpy
import random
from statistics import mean, stdev, pstdev


class NeuralNetworkModel:
    def __init__(self, genome={"weights": [1, 1, 1, 1, 1, 1, 1, 1], "biases": [0, 0, 0, 0, 0, 0, 0, 0, ]}):
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
    print("Average score {}".format(fitness))

    return fitness


def initialize_populatin(pop_size):
    population = []

    for _ in range(pop_size):
        genome = {"weights": [], "biases": []}
        for chromosome in genome:
            for i in range(8):
                genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
        population.append({"genome": genome})
    return population


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    ## Configuration values
    ant_simulations = 20
    simulation_max_steps = 350
    pop_size = 50
    genome_value_min = -1
    genome_value_max = 1

    ## Init population
    population = initialize_populatin(pop_size)

    ### Run stuff (evaluate population)
    for individual in population:
        evaluate(individual)

    best = max(population, key=lambda individual: individual["fitness"])
    print("best Individual was:   {}".format(best))

    average_population_fitness = mean([individual["fitness"] for individual in population])

    print("average population fitness:   {}".format(average_population_fitness))

    env.close()
