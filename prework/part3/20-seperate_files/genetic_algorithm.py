

import gym
import numpy
import random
from statistics import mean, stdev, pstdev
from itertools import count
from copy import deepcopy
import time
from dataclasses import dataclass, field
from os import path, makedirs
import csv
import json
from dacite import from_dict
from genome import Genome, create_new_genome
from config import Config

@dataclass
class Individ:
    id: int
    genome: Genome
    fitness = None


def create_new_individual(config:Config):
    # genome = {"weights": [1, random.random(), 1, random.random(), 0, 1, random.random(), 1],
    #           "biases": [0, random.random(), 0, 0, 2, random.random(), 0, 0, ]}
    id = next(config.global_individ_id_counter)
    genome = create_new_genome(id, config= config)
    individual = Individ(id=id, genome=genome)
    return individual


def initialize_populatin(config: Config):
    population = []
    for _ in range(config.pop_size):
        population.append(create_new_individual(config=config))
    return population


def replenish_population_with_new_random(pop_size, population):
    individuals_killed = pop_size - len(population)
    for _ in range(individuals_killed):
        individ = create_new_individual()
        population.append(individ)

    return population


def mutate(individual,config:Config):
    genome = individual.genome
    for chromosome_key, chromosome in vars(genome).items():
        # new_chromosome = []
        # if isinstance(genome.chromosomes[chromosome], list):
        for key in chromosome:
            chromosome[key].mutate(config=config)


def crossover(parent1, parent2, config:Config):
    assert parent1 != parent2
    # Setup

    if parent1.fitness < parent2.fitness:
        parent1, parent2 = deepcopy(parent2), deepcopy(parent1)

    p1_genome = parent1.genome
    p2_genome = parent1.genome
    id = next(config.global_individ_id_counter)
    child = Individ(id=id, genome=Genome())
    # Mix the genes
    chromosmes_that_can_mix_values = ["weight_chromosome", "bias_chromosome"]

    # crossover_neruons
    for chromosome_key, parent1_chromsome in vars(p1_genome).items():
        parent2_chromsome = p2_genome.get(chromosome_key)
        child_chromsome = deepcopy(parent1_chromsome)
        # Cross over chromoome
        # for gene_id in range(len(child_chromsome)):
        for id in child_chromsome:
            # parent1_gene_list = child_chromsome[id]
            # parent2_gene_list = parent2_chromsome[id]
            # for i, p1_gene in enumerate(parent1_gene_list):
            if id in parent2_chromsome:
                crossover_event = random.random()
                if child_chromsome[id].can_average_values_in_crossover:
                    if chromosome_key in chromosmes_that_can_mix_values and crossover_event < config.crossover_mix_genes_rate:
                        child_chromsome[id].value = (parent1_chromsome[id].value + parent2_chromsome[id].value) * 0.5
                    elif crossover_event < config.crossover_mix_genes_rate + (1 - config.crossover_mix_genes_rate) / 2:
                        child_chromsome[id] = parent2_chromsome[id]
                    ##else keep parent 1
                else:
                    if crossover_event < 0.5:
                        child_chromsome[id] = parent2_chromsome[id]
                    ##else keep parent 1

        child.genome.__setattr__(chromosome_key, child_chromsome)
    return child

