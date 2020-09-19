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



if __name__ == '__main__':
    # Configuration

    env = gym.make('LunarLander-v2')
    # neuralNetwork = NerualNetworkModel(24,4)
    ant_simulations = 10
    simulation_max_timesteps = 250
    population = []
    pop_size = 10

    ## Init population
    for i in range(pop_size):
        population.append({"weights": [random.random(), 1, 1, 1, 1, 1, 1, 1], "biases": [random.random(), 0, 0, 0, 0, 0, 0, 0]})

    ### Running stuff

    perceptron = Perceptron()

    for individual in population:
        total_score = 0

        for i_episode in range(ant_simulations):
            observation = env.reset()
            reward_in_current_simulation = 0

            for t in range(simulation_max_timesteps):
                # env.render()
                # print(observation)
                action = perceptron.run(observation)
                # print(action)
                observation, reward, done, info = env.step(action)
                reward_in_current_simulation += reward
                # print(reward_in_current_simulation)
                if done:
                    print("Episode finished after {} timesteps. Reward: {}".format(t + 1, reward_in_current_simulation))
                    total_score += reward_in_current_simulation
                    break
        # print("min_outputs {}".format(min_outputs))
        # print("max_outputs {}".format(max_outputs))
        print("Individual {} got an average score {}".format( individual,total_score / ant_simulations))
        env.close()
