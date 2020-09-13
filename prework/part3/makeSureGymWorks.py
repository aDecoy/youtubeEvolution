import gym
import numpy


def perceptron(input_values):
    weights = numpy.array([1, 1, 1.5, 1])
    biases =  numpy.array([0, 0, 0, 0])

    signal_strength = input_values * weights + biases
    output_signal = signal_strength.sum()

    if output_signal > 0 :
        return 1
    else:
        return 0

    return signal_strength




if __name__ == '__main__':

    env = gym.make('BipedalWalker-v2')
    total_score = 0

    ant_simulations = 50

    for i_episode in range(ant_simulations):
        observation = env.reset()
        for t in range(250):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            # action = perceptron(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                total_score += t
                break

    print("Average score {}".format(total_score/ant_simulations))
    env.close()
