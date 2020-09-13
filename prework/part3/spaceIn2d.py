import gym
import numpy
import torch

torch.no_grad()

ant_input,ant_output = 8,1

class Perceptron():
    def __init__(self,weights,biases ):
        weights = numpy.array([1] * 8)
        biases = numpy.array([0] * 8)
        self.weights = weights
        self.biases = biases

    def run(self,input_observations):
        signal_strength = input_observations * self.weights + self.biases
        output_signal = signal_strength.sum()

        if output_signal < 1:
            return 0
        elif output_signal < 2:
            return 1
        elif output_signal < 3:
            return 2
        elif output_signal < 4:
            return 3
        else:
            return 4

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # neuralNetwork = NerualNetworkModel(24,4)
    total_score = 0

    ant_simulations = 10

    # action = perceptron([0.45648593, 0.5706514, 0.91768444, -1.1972203, -0.38055214, -0.3717676,
    #                      0., 0.])
    perceptron = Perceptron(0,0)
    for i_episode in range(ant_simulations):
        observation = env.reset()
        for t in range(250):
            env.render()
            print(observation)
            print(len(observation))
            # action = env.action_space.sample()
            # for i in range(4):
            #     max_outputs[i] = max(max_outputs[i],action[i])
            #     min_outputs[i] = min(min_outputs[i],action[i])
            # print(action)
            action = perceptron.run(observation)
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                total_score += reward
                break
    # print("min_outputs {}".format(min_outputs))
    # print("max_outputs {}".format(max_outputs))
    print("Average score {}".format(total_score / ant_simulations))
    env.close()
