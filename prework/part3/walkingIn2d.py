import gym
import numpy
import torch

torch.no_grad()


def perceptron(input_values):
    weights = numpy.array([1 ]*24)
    biases =  numpy.array([0]*24)

    signal_strength = input_values * weights + biases
    output_signal = signal_strength.sum()

    if output_signal > 0 :
        return 1
    else:
        return 0

    return signal_strength

class NerualNetworkModel(torch.nn.Module):
    def __init__(self, ant_input_nodes,  ant_output_nodes):
        super(NerualNetworkModel, self).__init__()
        self.linear1 = torch.nn.Linear(ant_input_nodes, ant_output_nodes)

    def forward(self, x):
        output_signals = self.linear1(x).clamp(min=-1,max=1)
        return output_signals


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    neuralNetwork = NerualNetworkModel(24,4)
    total_score = 0

    ant_simulations = 10
    # action = [-0., - 0. ,- 0  , 0]
    # action = [0., 1 , 1  , 0]

    max_outputs= [0,0,0,0]
    min_outputs= [0,0,0,0]
    for i_episode in range(ant_simulations):
        observation = env.reset()
        for t in range(250):
            env.render()
            # print(observation)
            # print(len(observation))
            # action = env.action_space.sample()
            # for i in range(4):
            #     max_outputs[i] = max(max_outputs[i],action[i])
            #     min_outputs[i] = min(min_outputs[i],action[i])
            # print(action)
            actions_tensor = neuralNetwork(torch.Tensor(observation))
            actions = actions_tensor.tolist()
            print(actions)
            observation, reward, done, info = env.step(actions)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                total_score += reward
                break
    # print("min_outputs {}".format(min_outputs))
    # print("max_outputs {}".format(max_outputs))
    print("Average score {}".format(total_score/ant_simulations))
    env.close()
