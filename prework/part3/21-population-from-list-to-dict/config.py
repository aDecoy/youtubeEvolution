from dataclasses import dataclass
from itertools import count
from os import path, makedirs

import gym

from speciation import GenomeDistanceCashe


@dataclass
class Config:
    env_name = 'BipedalWalker-v3'
    ## Configuration values
    # Genome config
    genome_value_min = -1
    genome_value_max = 1
    # ant_hidden_layers = 0
    ant_input_neruons = 24
    ant_output_neruons = 4
    layer_sizes = [ant_input_neruons, 12, 4, ant_output_neruons]
    # Simulation run config
    simulation_max_steps = 500
    generations = 20
    pop_size = 10
    ant_simulations = 5
    observation_noise_std = 0.05
    use_observation_noise = False
    use_action_noise = False
    action_noise_rate = 0.05
    # GA params
    mutation_rate = 0.05
    mutation_power = 0.4
    crossover_mix_genes_rate = 0.1
    parent_selection_weighted_choises = False
    # speciation params
    gene_compatibility_disjoint_coefficient = 1.
    compatibility_weight_coefficient = 0.5
    compatibility_bias_coefficient = 0.5
    # How similar a genes has to be to be in same species
    compatibility_threshold = 4.5
    genomeDistanceCashe = GenomeDistanceCashe()
    species = {}

    # Playback settings
    # save_best_each = 2
    playback_folder = "playback-18-1"
    save_best_individual_folder = playback_folder + "/bestIndividuals"
    save_best_individual_base = save_best_individual_folder + "/bestIndividualInGeneration_"
    save_score_file_folder = playback_folder + "/fitnesses"
    save_score_file = save_score_file_folder + "/results.csv"

    global_individ_id_counter = count()
    global_bias_id_counter = count()
    global_weight_id_counter = count()


    def __post_init__(self):
        self.global_neruon_id_counter = count(start=self.ant_output_neruons)
        if not path.exists(self.playback_folder):
            makedirs(self.playback_folder)
            makedirs(self.save_best_individual_folder)
            makedirs(self.save_score_file_folder)
