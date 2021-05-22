from genetic_algorithm import Individ
from phenotype import develop_genome_to_neural_network_phenotype, NeuralNetworkModel
import numpy
import gym

class World:

    def __init__(self, config ):
        self.env =   gym.make(config.env_name)
        self.config=config


    def evaluate(self, individual: Individ, ant_simulations, render: bool = False):
        total_score = []
        # perceptron = NeuralNetworkModel(individual.genome)
        phenotype = develop_genome_to_neural_network_phenotype(individual.genome)  # todo mix these ? change function  name?
        # phenotype = develop_genome_to_neural_network_phenotype(geneB.genome)
        nerualNetwork = NeuralNetworkModel(phenotype, config=self.config)
        # perceptron = NeuralNetworkModel(gene.genome)
        for i_episode in range(ant_simulations):
            observation = self.env.reset()
            score_in_current_simulation = 0
            frozen_steps = 0
            previous_observation = [12345]
            for t in range(self.config.simulation_max_steps):
                if render:
                    self.env.render()
                # print(observation)
                # action = env.action_space.sample()
                if self.config.use_observation_noise:
                    observation = self.config.noizify_observations(observation)
                action = nerualNetwork.run(observation)
                # if g > -1 :
                #     print(action)
                observation, reward, done, info = self.env.step(action)
                if (observation == previous_observation).all():
                    frozen_steps += 1
                    if frozen_steps > 5:
                        done = True
                        frozen_steps = 0

                else:
                    frozen_steps = 0
                previous_observation = observation
                score_in_current_simulation += reward
                if done:
                    if render:
                        print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation))
                    total_score.append(score_in_current_simulation)
                    break

        fitness = sum(total_score) / ant_simulations
        # fitness = min(total_score)
        individual.fitness = fitness
        # print("Average score for individual {}:  {}".format(individual.id,fitness))

        return fitness

    def noizify_observations(self, observations):
        noise_mean = 0
        noise = numpy.random.normal(loc=noise_mean,
                                    scale=self.config.observation_noise_std,
                                    size=self.config.ant_input_neruons)
        observations = observations + noise
        return observations




