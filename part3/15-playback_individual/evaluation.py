import gym
from config import Config
from phenotype import NeuralNetworkModel, develop_genome_to_phenotype
import numpy
from statistics import  mean


def noizify_observations(observations, config):
    max_observation_noise = 0.2
    noise_mean = 0
    noise = numpy.random.normal(loc=noise_mean, scale=config.observation_noise_std, size=8)
    observations = observations + noise
    return observations


class World:

    def __init__(self, config: Config):
        self.config = config
        self.env = gym.make(config.env_name)

    def evaluate(self,individual, render = False ):
        total_score = []
        phenotype = develop_genome_to_phenotype(individual["genome"])
        neuralNetwork = NeuralNetworkModel(phenotype, config=self.config)
        for i_episode in range(self.config.ant_simulations):
            observation = self.env.reset()
            score_in_current_simulation = 0
            for t in range(self.config.simulation_max_steps):
                if render:
                    self.env.render()
                # print(observation)
                action = self.env.action_space.sample()
                if self.config.use_observation_noise:
                    observation = noizify_observations(observation)
                action = neuralNetwork.run(observation)
                # print(action)
                observation, reward, done, info = self.env.step(action)
                score_in_current_simulation += reward
                # print("score_in_current_simulation {}".format(score_in_current_simulation))
                # print("reward {}".format(reward))
                if done or t == self.config.simulation_max_steps-1:
                    if render:
                        print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation ))
                    total_score.append(score_in_current_simulation)
                    break

        # fitness = sum(total_score) / ant_simulations
        fitness = mean(total_score)
        assert len(total_score) == self.config.ant_simulations
        individual["fitness"] = fitness
        # print("Average score for individual {}".format(fitness))

        return fitness