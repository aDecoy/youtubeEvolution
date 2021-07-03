import gym
from config import Config
from phenotype import NeuralNetworkModel, develop_genome_to_phenotype
import numpy
from statistics import  mean, stdev
import multiprocessing as mp
from itertools import chain
import time

gym.logger.set_level(40)

def noizify_observations(observations, config):
    max_observation_noise = 0.2
    noise_mean = 0
    noise = numpy.random.normal(loc=noise_mean, scale=config.observation_noise_std, size=8)
    observations = observations + noise
    return observations

def eval_chunk(dict_params):
    config = dict_params["config"]
    chunk = dict_params["chunk"]
    chunk_id = dict_params["chunk_id"]
    print("starting chunk {}".format(chunk_id))
    # t0 = time.time()
    world = World(config=config)
    # t1 = time.time()
    # print("Time spent on just creating the world object {}".format(t1-t0))
    for individual in chunk:
        world.evaluate(individual=individual)
    print("done with chunk {}".format(chunk_id))
    return chunk



def evaluate(population: list, config: Config):

    # t0 = time.time()
    if not config.calculate_in_parallel:
        world = World(config=config)
        for individual in population:
            world.evaluate(individual)
        # t1 = time.time()
    else:
        results_of_chunks = []
        def log_result(result):
            results_of_chunks.append(result)

        number_of_chunks = config.number_of_chunks
        pool = mp.Pool(number_of_chunks)
        chunks = [list(population)[i::number_of_chunks] for i in range(number_of_chunks)]

        for chunk_id, chunk in enumerate(chunks):
            pool.apply_async(eval_chunk, args=({"chunk": chunk, "chunk_id": chunk_id, "config": config},), callback=log_result)
        # [pool.apply_async(eval_chunk,args= ({"chunk":chunk, "chunk_id": chunk_id, "config": config}, ), callback=log_result) for chunk_id, chunk in enumerate(chunks)]
        pool.close()
        pool.join()
        population = list(chain.from_iterable(results_of_chunks))

    # t2 = time.time()
    # print("Sync {}, async/parallel {}".format(t1-t0, t2-t1))
    population = { individual.id : individual for individual in population}
    return population


class World:
    def __init__(self, config: Config):
        self.config = config
        self.env = gym.make(config.env_name)

    def evaluate(self,individual, render = False ):
        total_score = []
        # total_timesteps = []
        phenotype = develop_genome_to_phenotype(individual.genome)
        neuralNetwork = NeuralNetworkModel(phenotype, config=self.config)
        for i_episode in range(self.config.ant_simulations):
            observation = self.env.reset()
            score_in_current_simulation = 0
            for t in range(self.config.simulation_max_steps):
                # render=True
                if render:
                    self.env.render()
                # print(observation)
                # action = self.env.action_space.sample()
                if self.config.use_observation_noise:
                    observation = noizify_observations(observation)
                action = neuralNetwork.run(observation)
                # print(action)
                observation, reward, done, info = self.env.step(action)
                # print(observation)
                score_in_current_simulation += reward
                # print("score_in_current_simulation {}".format(score_in_current_simulation))
                # print("reward {}".format(reward))
                if done or t == self.config.simulation_max_steps-1:
                    if render:
                        print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation ))
                    total_score.append(score_in_current_simulation)
                    # total_timesteps.append(t + 1)
                    break

        # fitness = sum(total_score) / ant_simulations
        fitness = mean(total_score)
        assert len(total_score) == self.config.ant_simulations
        individual.fitness = fitness
        # print("Average score for individual {}".format(fitness))
        # print("std for individual fitness evaluation {}".format(stdev(total_score)))
        # print(total_score)
        # print("Average timesteps for individual {}".format(mean(total_timesteps)))
        return fitness