import gym
import numpy


def perceptron(input_values):
    weights = numpy.array([1, 1, 1, 1,1, 1, 1, 1])
    biases =  numpy.array([0, 0, 0, 0, 0, 0, 0, 0,])

    signal_strength = input_values * weights + biases
    output_signal = signal_strength.sum()

    if output_signal < 0.25 :
        return 0
    elif output_signal < 0.50:
        return 1
    elif output_signal < 0.75:
        return 2
    else: # 0.75 to 1
        return 3

    return signal_strength

if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    total_score = 0

    ant_simulations = 50

    for i_episode in range(ant_simulations):
        observation = env.reset()
        for t in range(250):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            print(action)
            action = perceptron(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                total_score += t
                break

    print("Average score {}".format(total_score/ant_simulations))
    env.close()
