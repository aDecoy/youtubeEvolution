import gym
import numpy
import random


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
    perceptron = NeuralNetworkModel(individual)
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


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    ## Configuration values
    ant_simulations = 20
    simulation_max_steps = 350
    pop_size = 100

    ## Init population
    population = []
    for _ in range(pop_size):
        population.append({
            "genome": {"weights": [1, random.random(), 1, random.random(), 0, 1, random.random(), 1],
                       "biases": [0, random.random(), 0, 0, 2, random.random(), 0, 0]},
            "fitness": 0})

    ### Run stuff (evaluate population)
    for individual in population:
        evaluate(individual)

    best = max(population, key=lambda individual: individual["fitness"])
    print("best  {}".format(best))

    env.close()
