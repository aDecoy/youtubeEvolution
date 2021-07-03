import random
from config import Config
from copy import deepcopy
from genome import Genome, create_new_genome



def create_new_individual(config: Config):
    genome = create_new_genome(config=config)
    # genome = {"weights": [], "biases": []}
    # for chromosome in genome:
    #     for i in range(8):
    #         genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    #     # genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    individual = {"id": next(config.global_individ_id_counter), "genome": genome}
    return individual


def initialize_population(config: Config):
    population = []
    for _ in range(config.pop_size):
        population.append(create_new_individual(config=config))
    return population


def replenish_population_with_new_random(config : Config, population : list):
    individuals_killed = config.pop_size - len(population)
    for _ in range(individuals_killed):
        individ = create_new_individual(config= config)
        population.append(individ)

    return population


def mutate(individual, config: Config):
    genome = individual["genome"]
    for chromosome_key, chromosome in vars(genome).items():
        for gene_key in chromosome:
            if random.random() < config.mutation_rate:
                chromosome[gene_key].mutate()


def crossover(parent1: dict, parent2: dict, config: Config):
    assert parent1 != parent2

    if parent1["fitness"] < parent2["fitness"]:
        parent1, parent2 = deepcopy(parent2), deepcopy(parent1)

    # Setup
    p1_genome = parent1["genome"]
    p2_genome = parent1["genome"]
    child = {"id": next(config.global_individ_id_counter), "genome": Genome()}
    # Mix the genes

    chromosomes_that_can_average_values = ["weight_chromosome", "bias_chromosome"]

    # crossover neruons
    for chromosome_key, parent1_chromosome in vars(p1_genome).items():
        parent2_chromosome = p2_genome.__getattribute__(chromosome_key)

        child_chromsome = deepcopy(parent1_chromosome)
        for gene_id in child_chromsome:
            if gene_id in parent2_chromosome:
                # cross over
                crossover_event = random.random()
                if crossover_event < config.crossover_mix_genes_rate and chromosome_key in chromosomes_that_can_average_values:
                    child_chromsome[gene_id] = (parent1_chromosome[gene_id].value + parent2_chromosome[gene_id].value) * 0.5
                elif crossover_event < config.crossover_mix_genes_rate + (1 - config.crossover_mix_genes_rate) / 2:
                    # keep gene from parent 2
                    child_chromsome[gene_id] = parent2_chromosome[gene_id]
            else:
                # keep gene from parent 1
                pass

        child["genome"].__setattr__(chromosome_key, child_chromsome)

    return child
