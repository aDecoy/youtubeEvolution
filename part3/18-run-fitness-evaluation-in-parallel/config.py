from dataclasses import dataclass
from itertools import count
from os import path, makedirs
from shutil import  rmtree, copyfile

@dataclass
class Config:
    env_name = 'BipedalWalker-v3'
    ## Configuration values
    # Genome config
    genome_value_min = -1
    genome_value_max = 1
    ant_input_neurons = 24
    ant_output_neurons = 4
    layer_sizes = [ant_input_neurons, 6, 6,4, ant_output_neurons]
    # Simulation run config
    simulation_max_steps = 1600
    generations = 30
    pop_size = 200
    # ant_simulations = 15
    ant_simulations = 15
    calculate_in_parallel = True
    number_of_chunks = 3
    observation_noise_std = 0.05
    use_observation_noise = False
    use_action_noise = False
    action_noise_rate = 0.1
    # GA params
    mutation_rate = 0.05
    mutation_power = 0.4
    parent_rate = 0.10
    survival_rate = 0.10
    crossover_mix_genes_rate = 0.1
    parent_selection_weighted_choises = False

    global_individ_id_counter = count()
    global_weight_chromsome_id_counter = count()
    global_bias_chromsome_id_counter = count()
    global_neuron_chromsome_id_counter = count(start=layer_sizes[-1])


    # Playback settings
    playback_folder = "playback-18-4"
    saved_best_individual_per_generation_folder = playback_folder + "/bestIndividuals"
    saved_best_individual_base = playback_folder + "/best_Individual_in_generation_"
    saved_fitness_history_file = playback_folder + "/fitness_history.csv"




    def __post_init__(self):
        if not path.exists(self.playback_folder):
            makedirs(self.playback_folder)

        copyfile("config.py", self.playback_folder+"/config.py")

