import gym
import numpy
import random
from statistics import mean, stdev, pstdev, median
from itertools import count
from copy import deepcopy
from collections import Counter
import time
from dataclasses import dataclass

global_individ_id_counter = count()
global_weight_chromsome_id_counter = count()
global_bias_chromsome_id_counter = count()


@dataclass
class Genome:
    layer_chromosome: dict = None
    neuron_chromosme: dict = None
    weight_chromosme: dict = None
    bias_chromosme: dict = None


@dataclass
class BiasGene:
    innovationNumber: int
    value: float
    parent_neuron: int

    def mutate(self):
        self.value += random.uniform(-mutation_power, +mutation_power)


@dataclass
class WeightGene:
    innovationNumber: int
    value: float
    transmitting_neuron: int
    receiving_neuron: int

    def mutate(self):
        self.value += random.uniform(-mutation_power, +mutation_power)


@dataclass
class NeruonGene:
    innovationNumber: int
    parent_layer: int

    def mutate(self):
        pass


@dataclass
class LayerGene:
    innovationNumber: int

    # later add activation function
    def mutate(self):
        pass


def create_new_layer_chromosome(ant_layers):
    return {i: LayerGene(innovationNumber=i) for i in range(ant_layers)}


def create_new_neruon_chromosome(layer_sizes):
    neurons = {}
    input_innovation_number = -1

    # Input neurons
    for i in range(layer_sizes[0]):
        neurons[input_innovation_number] = NeruonGene(innovationNumber=input_innovation_number, parent_layer=0)
        input_innovation_number -= 1

    # Hidden neurons
    for layer, ant_neurons in enumerate(layer_sizes[1:-1], start=1):
        for i in range(ant_neurons):
            id = next(global_neuron_chromsome_id_counter)
            neurons[id] = NeruonGene(innovationNumber=id, parent_layer=layer)
    # Output neruon
    output_innovation_number = 0
    for i in range(layer_sizes[-1]):
        id = output_innovation_number
        output_innovation_number += 1
        neurons[id] = NeruonGene(innovationNumber=id, parent_layer=len(layer_sizes) - 1)

    return neurons


def create_new_weight_chromosome(neuron_choromosome: dict, ant_layers: int):
    weights = {}
    # input neurons
    transmitting_neurons = {key: gene for key, gene in neuron_choromosome.items() if gene.parent_layer == 0}

    # Loop over the rest of the layers, finding values for all weights
    for receiving_layer in range(1, ant_layers):
        receiving_neurons = {key: gene for key, gene in neuron_choromosome.items() if gene.parent_layer == receiving_layer}

        # for each gene
        for transmitting_neuron in transmitting_neurons:
            for receiving_neuron in receiving_neurons:
                id = next(global_weight_chromsome_id_counter)
                weights[id] = WeightGene(innovationNumber=id, transmitting_neuron=transmitting_neuron, receiving_neuron=receiving_neuron,
                                         value=random.uniform(genome_value_min, genome_value_max))

        transmitting_neurons = receiving_neurons

    return weights


def create_new_bias_chromosome(neuron_chromosome):
    biases = {}
    for neuron_id in neuron_chromosome:
        if neuron_id >= 0:
            biases[neuron_id] = BiasGene(innovationNumber=neuron_id, parent_neuron=neuron_id,
                                         value=random.uniform(genome_value_min, genome_value_max))
    return biases


def create_new_genome(layer_sizes):
    layers = create_new_layer_chromosome(len(layer_sizes))
    neurons = create_new_neruon_chromosome(layer_sizes)
    weights = create_new_weight_chromosome(neurons, ant_layers=len(layer_sizes))
    biases = create_new_bias_chromosome(neuron_chromosome=neurons)

    return Genome(layer_chromosome=layers,
                  neuron_chromosme=neurons,
                  weight_chromosme=weights,
                  bias_chromosme=biases)


@dataclass
class Phenotype:
    weights: dict
    biases: dict


def develop_genome_to_phenotype(genome: Genome):
    ant_layers = len(genome.layer_chromosome)
    biases = {}
    for receiving_layer in range(1, ant_layers):
        receiving_neurons = [key for (key, neuron) in genome.neuron_chromosme.items() if neuron.parent_layer == receiving_layer]
        biases[receiving_layer] = numpy.array([genome.bias_chromosme[key].value for key in receiving_neurons])

    weights = {}
    transmitting_neurons = [key for (key, neuron) in genome.neuron_chromosme.items() if neuron.parent_layer == 0]
    for receiving_layer in range(1, ant_layers):
        receiving_neurons = [key for (key, neuron) in genome.neuron_chromosme.items() if neuron.parent_layer == receiving_layer]
        layer_weights = []
        for receiving_neuron in receiving_neurons:
            layer_weights.append([weight_gene.value for weightKey, weight_gene in genome.weight_chromosme.items()
                                  if (weight_gene.receiving_neuron == receiving_neuron and
                                      weight_gene.transmitting_neuron in transmitting_neurons)])

        weights["{}-{}".format(receiving_layer - 1, receiving_layer)] = numpy.array(layer_weights)
        transmitting_neurons = receiving_neurons

    return Phenotype(weights=weights, biases=biases)


class NeuralNetworkModel:

    # def __init__(self, genome={"weights": [[2, 0.9, 1.2, 1.1, 1, 1, 1, 1]], "biases": [0, 0, 0, 0, 0, 0, 0, 0, ]}):
    def __init__(self, phenotype: Phenotype):
        # 8 input, 4 hidden , 1 output
        # if genome is None:

        self.weights = phenotype.weights
        self.biases = phenotype.biases

    def noise_output(self):
        return random.choice(range(3))

    def run(self, input_values):

        if use_action_noise:
            if random.random() < action_noise_rate:
                return self.noise_output()

        input_values = numpy.array(input_values).reshape(-1)
        layer_outputs = [input_values]
        for i in range(1,len(layer_sizes)):
            layer_outputs.append( numpy.matmul(self.weights["{}-{}".format(i-1,i)], layer_outputs[i-1]) + self.biases[i])

        best_action = numpy.argmax(layer_outputs[-1])
        return best_action


def evaluate(individual):
    total_score = []
    # phenotype = develop_genome_to_phenotype(individual["genome"])
    phenotype = develop_genome_to_phenotype(geneB)
    neuralNetwork = NeuralNetworkModel(phenotype)
    # perceptron = NeuralNetworkModel(gene["genome"])
    for i_episode in range(ant_simulations):
        observation = env.reset()
        score_in_current_simulation = 0
        for t in range(simulation_max_steps):
            env.render()
            # print(observation)
            # action = env.action_space.sample()
            if use_observation_noise:
                observation = noizify_observations(observation)
            action = neuralNetwork.run(observation)
            # print(action)
            observation, reward, done, info = env.step(action)
            score_in_current_simulation += reward
            # print("score_in_current_simulation {}".format(score_in_current_simulation))
            # print("reward {}".format(reward))
            if done or t == simulation_max_steps-1:
                print("Episode finished after {} timesteps, score {}".format(t + 1, score_in_current_simulation ))
                total_score.append(score_in_current_simulation)
                break

    # fitness = sum(total_score) / ant_simulations
    fitness = mean(total_score)
    assert len(total_score) == ant_simulations
    individual["fitness"] = fitness
    # print("Average score for individual {}".format(fitness))

    return fitness


def create_new_individual():
    genome = create_new_genome(layer_sizes=layer_sizes)
    # genome = {"weights": [], "biases": []}
    # for chromosome in genome:
    #     for i in range(8):
    #         genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    #     # genome[chromosome].append(random.uniform(genome_value_min, genome_value_max))
    individual = {"id": next(global_individ_id_counter), "genome": genome}
    return individual


def initialize_populatin(pop_size):
    population = []
    for _ in range(pop_size):
        population.append(create_new_individual())
    return population


def replenish_population_with_new_random(pop_size, population):
    individuals_killed = pop_size - len(population)
    for _ in range(individuals_killed):
        individ = create_new_individual()
        population.append(individ)

    return population


def mutate(individual):
    genome = individual["genome"]
    for chromosome_key, chromosome in vars(genome).items():
        for gene_key in chromosome:
            if random.random() < mutation_rate:
                chromosome[gene_key].mutate()


def crossover(parent1: dict, parent2: dict):
    assert parent1 != parent2

    if parent1["fitness"] < parent2["fitness"]:
        parent1, parent2 = deepcopy(parent2), deepcopy(parent1)

    # Setup
    p1_genome = parent1["genome"]
    p2_genome = parent1["genome"]
    child = {"id": next(global_individ_id_counter), "genome": Genome()}
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
                if crossover_event < crossover_mix_genes_rate and chromosome_key in chromosomes_that_can_average_values:
                    child_chromsome[gene_id] = (parent1_chromosome[gene_id].value + parent2_chromosome[gene_id].value) * 0.5
                elif crossover_event < crossover_mix_genes_rate + (1 - crossover_mix_genes_rate) / 2:
                    # keep gene from parent 2
                    child_chromsome[gene_id] = parent2_chromosome[gene_id]
            else:
                # keep gene from parent 1
                pass

        child["genome"].__setattr__(chromosome_key, child_chromsome)

    return child


def noizify_observations(observations):
    max_observation_noise = 0.2
    noise_mean = 0
    noise = numpy.random.normal(loc=noise_mean, scale=observation_noise_std, size=8)
    observations = observations + noise
    return observations


if __name__ == '__main__':
    # todo : look for bottle necks

    env = gym.make('LunarLander-v2')
    ## Configuration values
    # Genome config
    genome_value_min = -1
    genome_value_max = 1
    ant_input_neurons = 8
    ant_output_neurons = 4
    layer_sizes = [ant_input_neurons, 6,6,4, ant_output_neurons]
    # Simulation run config
    simulation_max_steps = 380
    generations = 50
    pop_size = 100
    # ant_simulations = 15
    ant_simulations = 30
    observation_noise_std = 0.05
    use_observation_noise = False
    use_action_noise = False
    action_noise_rate = 0.1
    # GA params
    mutation_rate = 0.05
    mutation_power = 0.4
    parent_rate = 0.1
    survival_rate = 0.10
    crossover_mix_genes_rate = 0.1
    parent_selection_weighted_choises = False


    global_neuron_chromsome_id_counter = count(start=layer_sizes[-1])

    ## Init population
    population = initialize_populatin(pop_size)

    ### Run stuff (evolve the population to find the best individual)
    # gene =  {'id': 34595, 'genome': {'weights': [1.8279610414408418, -1.215151552382231, -1.914386503678685, -1.5992229022420927, -0.4629710592800763, 0.09949141104419473, -0.8159553270867125, 0.3772991566011524], 'biases': [-2.1137517590961803, 0.23877397428813674, 0.6899723535739337, 0.20582457309766966, 1.8034607671721203, -0.9274868121058241, -0.3656953731942547, -0.009541580549235956]}, 'fitness': 16.002701150260478}

    # geneB = Genome(layer_chromosome={0: LayerGene(innovationNumber=0), 1: LayerGene(innovationNumber=1), 2: LayerGene(innovationNumber=2)}, neuron_chromosme={-1: NeruonGene(innovationNumber=-1, parent_layer=0), -2: NeruonGene(innovationNumber=-2, parent_layer=0), -3: NeruonGene(innovationNumber=-3, parent_layer=0), -4: NeruonGene(innovationNumber=-4, parent_layer=0), -5: NeruonGene(innovationNumber=-5, parent_layer=0), -6: NeruonGene(innovationNumber=-6, parent_layer=0), -7: NeruonGene(innovationNumber=-7, parent_layer=0), -8: NeruonGene(innovationNumber=-8, parent_layer=0), 216: NeruonGene(innovationNumber=216, parent_layer=1), 217: NeruonGene(innovationNumber=217, parent_layer=1), 218: NeruonGene(innovationNumber=218, parent_layer=1), 219: NeruonGene(innovationNumber=219, parent_layer=1), 0: NeruonGene(innovationNumber=0, parent_layer=2), 1: NeruonGene(innovationNumber=1, parent_layer=2), 2: NeruonGene(innovationNumber=2, parent_layer=2), 3: NeruonGene(innovationNumber=3, parent_layer=2)}, weight_chromosme={2544: WeightGene(innovationNumber=2544, value=-0.3149036725153034, transmitting_neuron=-1, receiving_neuron=216), 2545: WeightGene(innovationNumber=2545, value=0.9364857461599881, transmitting_neuron=-1, receiving_neuron=217), 2546: WeightGene(innovationNumber=2546, value=0.4018259264421451, transmitting_neuron=-1, receiving_neuron=218), 2547: WeightGene(innovationNumber=2547, value=1.233926365580472, transmitting_neuron=-1, receiving_neuron=219), 2548: WeightGene(innovationNumber=2548, value=-0.28138587135646365, transmitting_neuron=-2, receiving_neuron=216), 2549: WeightGene(innovationNumber=2549, value=-0.17974124766620314, transmitting_neuron=-2, receiving_neuron=217), 2550: WeightGene(innovationNumber=2550, value=-0.2415398394986331, transmitting_neuron=-2, receiving_neuron=218), 2551: WeightGene(innovationNumber=2551, value=2.2465225639234268, transmitting_neuron=-2, receiving_neuron=219), 2552: WeightGene(innovationNumber=2552, value=-0.18674906793802304, transmitting_neuron=-3, receiving_neuron=216), 2553: WeightGene(innovationNumber=2553, value=-0.7490965582882687, transmitting_neuron=-3, receiving_neuron=217), 2554: WeightGene(innovationNumber=2554, value=1.1191904482894277, transmitting_neuron=-3, receiving_neuron=218), 2555: WeightGene(innovationNumber=2555, value=-0.40379997491768194, transmitting_neuron=-3, receiving_neuron=219), 2556: WeightGene(innovationNumber=2556, value=0.7634938999679581, transmitting_neuron=-4, receiving_neuron=216), 2557: WeightGene(innovationNumber=2557, value=-0.47342457606208393, transmitting_neuron=-4, receiving_neuron=217), 2558: WeightGene(innovationNumber=2558, value=-0.8519535086590251, transmitting_neuron=-4, receiving_neuron=218), 2559: WeightGene(innovationNumber=2559, value=0.8897155626538398, transmitting_neuron=-4, receiving_neuron=219), 2560: WeightGene(innovationNumber=2560, value=-1.7884062383188037, transmitting_neuron=-5, receiving_neuron=216), 2561: WeightGene(innovationNumber=2561, value=0.2537179898382387, transmitting_neuron=-5, receiving_neuron=217), 2562: WeightGene(innovationNumber=2562, value=-0.8195492531198745, transmitting_neuron=-5, receiving_neuron=218), 2563: WeightGene(innovationNumber=2563, value=0.3137714007991538, transmitting_neuron=-5, receiving_neuron=219), 2564: WeightGene(innovationNumber=2564, value=-0.7430989075265061, transmitting_neuron=-6, receiving_neuron=216), 2565: WeightGene(innovationNumber=2565, value=-0.2453049535329691, transmitting_neuron=-6, receiving_neuron=217), 2566: WeightGene(innovationNumber=2566, value=-0.30770889689302555, transmitting_neuron=-6, receiving_neuron=218), 2567: WeightGene(innovationNumber=2567, value=1.8031398619285852, transmitting_neuron=-6, receiving_neuron=219), 2568: WeightGene(innovationNumber=2568, value=-0.9398702756445587, transmitting_neuron=-7, receiving_neuron=216), 2569: WeightGene(innovationNumber=2569, value=0.022005231971392925, transmitting_neuron=-7, receiving_neuron=217), 2570: WeightGene(innovationNumber=2570, value=0.7842291188597511, transmitting_neuron=-7, receiving_neuron=218), 2571: WeightGene(innovationNumber=2571, value=0.038206615766167074, transmitting_neuron=-7, receiving_neuron=219), 2572: WeightGene(innovationNumber=2572, value=0.5580649814679151, transmitting_neuron=-8, receiving_neuron=216), 2573: WeightGene(innovationNumber=2573, value=-0.4122351106452592, transmitting_neuron=-8, receiving_neuron=217), 2574: WeightGene(innovationNumber=2574, value=-0.3992915241141822, transmitting_neuron=-8, receiving_neuron=218), 2575: WeightGene(innovationNumber=2575, value=0.4719178033080408, transmitting_neuron=-8, receiving_neuron=219), 2576: WeightGene(innovationNumber=2576, value=-1.1695985829884767, transmitting_neuron=216, receiving_neuron=0), 2577: WeightGene(innovationNumber=2577, value=0.760029711870561, transmitting_neuron=216, receiving_neuron=1), 2578: WeightGene(innovationNumber=2578, value=-0.9212237723466279, transmitting_neuron=216, receiving_neuron=2), 2579: WeightGene(innovationNumber=2579, value=0.20056119429781105, transmitting_neuron=216, receiving_neuron=3), 2580: WeightGene(innovationNumber=2580, value=-1.5825236527293842, transmitting_neuron=217, receiving_neuron=0), 2581: WeightGene(innovationNumber=2581, value=-0.9342770968622511, transmitting_neuron=217, receiving_neuron=1), 2582: WeightGene(innovationNumber=2582, value=0.4565868435837896, transmitting_neuron=217, receiving_neuron=2), 2583: WeightGene(innovationNumber=2583, value=-0.294417921757956, transmitting_neuron=217, receiving_neuron=3), 2584: WeightGene(innovationNumber=2584, value=-1.3128551912105142, transmitting_neuron=218, receiving_neuron=0), 2585: WeightGene(innovationNumber=2585, value=0.8996417305193, transmitting_neuron=218, receiving_neuron=1), 2586: WeightGene(innovationNumber=2586, value=0.43414621585618746, transmitting_neuron=218, receiving_neuron=2), 2587: WeightGene(innovationNumber=2587, value=-0.8171131744219299, transmitting_neuron=218, receiving_neuron=3), 2588: WeightGene(innovationNumber=2588, value=0.3863570567822829, transmitting_neuron=219, receiving_neuron=0), 2589: WeightGene(innovationNumber=2589, value=0.6649327134869388, transmitting_neuron=219, receiving_neuron=1), 2590: WeightGene(innovationNumber=2590, value=-1.5977730794956981, transmitting_neuron=219, receiving_neuron=2), 2591: WeightGene(innovationNumber=2591, value=-0.667636234886857, transmitting_neuron=219, receiving_neuron=3)}, bias_chromosme={216: BiasGene(innovationNumber=216, value=-0.19519572283788783, parent_neuron=216), 217: BiasGene(innovationNumber=217, value=0.21375551422271122, parent_neuron=217), 218: BiasGene(innovationNumber=218, value=0.41439826334168406, parent_neuron=218), 219: BiasGene(innovationNumber=219, value=-0.10392500798731527, parent_neuron=219), 0: BiasGene(innovationNumber=0, value=0.7981481306771181, parent_neuron=0), 1: BiasGene(innovationNumber=1, value=0.1351790463782115, parent_neuron=1), 2: BiasGene(innovationNumber=2, value=-0.6792201719978745, parent_neuron=2), 3: BiasGene(innovationNumber=3, value=-1.143843633283339, parent_neuron=3)})
    # geneB = Genome(layer_chromosome={0: LayerGene(innovationNumber=0), 1: LayerGene(innovationNumber=1)}, neuron_chromosme={-1: NeruonGene(innovationNumber=-1, parent_layer=0), -2: NeruonGene(innovationNumber=-2, parent_layer=0), -3: NeruonGene(innovationNumber=-3, parent_layer=0), -4: NeruonGene(innovationNumber=-4, parent_layer=0), -5: NeruonGene(innovationNumber=-5, parent_layer=0), -6: NeruonGene(innovationNumber=-6, parent_layer=0), -7: NeruonGene(innovationNumber=-7, parent_layer=0), -8: NeruonGene(innovationNumber=-8, parent_layer=0), 0: NeruonGene(innovationNumber=0, parent_layer=1), 1: NeruonGene(innovationNumber=1, parent_layer=1), 2: NeruonGene(innovationNumber=2, parent_layer=1), 3: NeruonGene(innovationNumber=3, parent_layer=1)}, weight_chromosme={2944: WeightGene(innovationNumber=2944, value=0.387936985204036, transmitting_neuron=-1, receiving_neuron=0), 2945: WeightGene(innovationNumber=2945, value=0.00427271395561285, transmitting_neuron=-1, receiving_neuron=1), 2946: WeightGene(innovationNumber=2946, value=0.7477691883674286, transmitting_neuron=-1, receiving_neuron=2), 2947: WeightGene(innovationNumber=2947, value=-1.0410353578830145, transmitting_neuron=-1, receiving_neuron=3), 2948: WeightGene(innovationNumber=2948, value=-0.9230912625111106, transmitting_neuron=-2, receiving_neuron=0), 2949: WeightGene(innovationNumber=2949, value=0.3297782187513623, transmitting_neuron=-2, receiving_neuron=1), 2950: WeightGene(innovationNumber=2950, value=-0.39718991885471566, transmitting_neuron=-2, receiving_neuron=2), 2951: WeightGene(innovationNumber=2951, value=0.4847764549562135, transmitting_neuron=-2, receiving_neuron=3), 2952: WeightGene(innovationNumber=2952, value=0.8388313781804336, transmitting_neuron=-3, receiving_neuron=0), 2953: WeightGene(innovationNumber=2953, value=1.2828925551782824, transmitting_neuron=-3, receiving_neuron=1), 2954: WeightGene(innovationNumber=2954, value=0.17517869390701077, transmitting_neuron=-3, receiving_neuron=2), 2955: WeightGene(innovationNumber=2955, value=-1.4356974661638318, transmitting_neuron=-3, receiving_neuron=3), 2956: WeightGene(innovationNumber=2956, value=-0.8689759283707534, transmitting_neuron=-4, receiving_neuron=0), 2957: WeightGene(innovationNumber=2957, value=0.4752672093361613, transmitting_neuron=-4, receiving_neuron=1), 2958: WeightGene(innovationNumber=2958, value=-2.2723339403879725, transmitting_neuron=-4, receiving_neuron=2), 2959: WeightGene(innovationNumber=2959, value=0.995819817590483, transmitting_neuron=-4, receiving_neuron=3), 2960: WeightGene(innovationNumber=2960, value=0.6105687332886398, transmitting_neuron=-5, receiving_neuron=0), 2961: WeightGene(innovationNumber=2961, value=-1.7071390145653484, transmitting_neuron=-5, receiving_neuron=1), 2962: WeightGene(innovationNumber=2962, value=0.7760032698885464, transmitting_neuron=-5, receiving_neuron=2), 2963: WeightGene(innovationNumber=2963, value=0.9440886970456386, transmitting_neuron=-5, receiving_neuron=3), 2964: WeightGene(innovationNumber=2964, value=-0.49823713052136226, transmitting_neuron=-6, receiving_neuron=0), 2965: WeightGene(innovationNumber=2965, value=-1.807673764092652, transmitting_neuron=-6, receiving_neuron=1), 2966: WeightGene(innovationNumber=2966, value=1.3478779919692505, transmitting_neuron=-6, receiving_neuron=2), 2967: WeightGene(innovationNumber=2967, value=0.4184192322590505, transmitting_neuron=-6, receiving_neuron=3), 2968: WeightGene(innovationNumber=2968, value=0.5661499117576545, transmitting_neuron=-7, receiving_neuron=0), 2969: WeightGene(innovationNumber=2969, value=1.3049717125112408, transmitting_neuron=-7, receiving_neuron=1), 2970: WeightGene(innovationNumber=2970, value=0.9195057060575189, transmitting_neuron=-7, receiving_neuron=2), 2971: WeightGene(innovationNumber=2971, value=-0.3059734060873733, transmitting_neuron=-7, receiving_neuron=3), 2972: WeightGene(innovationNumber=2972, value=0.7304594976624686, transmitting_neuron=-8, receiving_neuron=0), 2973: WeightGene(innovationNumber=2973, value=0.3950747877505427, transmitting_neuron=-8, receiving_neuron=1), 2974: WeightGene(innovationNumber=2974, value=1.2096615661868464, transmitting_neuron=-8, receiving_neuron=2), 2975: WeightGene(innovationNumber=2975, value=-1.640827902467262, transmitting_neuron=-8, receiving_neuron=3)}, bias_chromosme={0: BiasGene(innovationNumber=0, value=0.2669646309138031, parent_neuron=0), 1: BiasGene(innovationNumber=1, value=0.16253782257024457, parent_neuron=1), 2: BiasGene(innovationNumber=2, value=-0.038590975984096, parent_neuron=2), 3: BiasGene(innovationNumber=3, value=0.23681087769761244, parent_neuron=3)})
    # geneB = Genome(layer_chromosome={0: LayerGene(innovationNumber=0), 1: LayerGene(innovationNumber=1)}, neuron_chromosme={-1: NeruonGene(innovationNumber=-1, parent_layer=0), -2: NeruonGene(innovationNumber=-2, parent_layer=0), -3: NeruonGene(innovationNumber=-3, parent_layer=0), -4: NeruonGene(innovationNumber=-4, parent_layer=0), -5: NeruonGene(innovationNumber=-5, parent_layer=0), -6: NeruonGene(innovationNumber=-6, parent_layer=0), -7: NeruonGene(innovationNumber=-7, parent_layer=0), -8: NeruonGene(innovationNumber=-8, parent_layer=0), 0: NeruonGene(innovationNumber=0, parent_layer=1), 1: NeruonGene(innovationNumber=1, parent_layer=1), 2: NeruonGene(innovationNumber=2, parent_layer=1), 3: NeruonGene(innovationNumber=3, parent_layer=1)}, weight_chromosme={1312: WeightGene(innovationNumber=1312, value=1.1654966661674453, transmitting_neuron=-1, receiving_neuron=0), 1313: WeightGene(innovationNumber=1313, value=-0.3395514092033732, transmitting_neuron=-1, receiving_neuron=1), 1314: WeightGene(innovationNumber=1314, value=1.549531385413664, transmitting_neuron=-1, receiving_neuron=2), 1315: WeightGene(innovationNumber=1315, value=-0.7678575271607081, transmitting_neuron=-1, receiving_neuron=3), 1316: WeightGene(innovationNumber=1316, value=0.23965717020467253, transmitting_neuron=-2, receiving_neuron=0), 1317: WeightGene(innovationNumber=1317, value=0.30914215489974634, transmitting_neuron=-2, receiving_neuron=1), 1318: WeightGene(innovationNumber=1318, value=-1.2765401083499235, transmitting_neuron=-2, receiving_neuron=2), 1319: WeightGene(innovationNumber=1319, value=-0.3855627818199777, transmitting_neuron=-2, receiving_neuron=3), 1320: WeightGene(innovationNumber=1320, value=-0.7155310736663647, transmitting_neuron=-3, receiving_neuron=0), 1321: WeightGene(innovationNumber=1321, value=0.4699640994189003, transmitting_neuron=-3, receiving_neuron=1), 1322: WeightGene(innovationNumber=1322, value=-0.7808474387393031, transmitting_neuron=-3, receiving_neuron=2), 1323: WeightGene(innovationNumber=1323, value=0.06062367821269227, transmitting_neuron=-3, receiving_neuron=3), 1324: WeightGene(innovationNumber=1324, value=0.7375786923798185, transmitting_neuron=-4, receiving_neuron=0), 1325: WeightGene(innovationNumber=1325, value=-0.3726856803416921, transmitting_neuron=-4, receiving_neuron=1), 1326: WeightGene(innovationNumber=1326, value=-1.1184222305029796, transmitting_neuron=-4, receiving_neuron=2), 1327: WeightGene(innovationNumber=1327, value=-0.4325111644431017, transmitting_neuron=-4, receiving_neuron=3), 1328: WeightGene(innovationNumber=1328, value=-0.089577497742103, transmitting_neuron=-5, receiving_neuron=0), 1329: WeightGene(innovationNumber=1329, value=-0.9535590637475367, transmitting_neuron=-5, receiving_neuron=1), 1330: WeightGene(innovationNumber=1330, value=-0.39579713555862106, transmitting_neuron=-5, receiving_neuron=2), 1331: WeightGene(innovationNumber=1331, value=-0.7050233060258909, transmitting_neuron=-5, receiving_neuron=3), 1332: WeightGene(innovationNumber=1332, value=1.7560761559164544, transmitting_neuron=-6, receiving_neuron=0), 1333: WeightGene(innovationNumber=1333, value=-1.1275039650662482, transmitting_neuron=-6, receiving_neuron=1), 1334: WeightGene(innovationNumber=1334, value=1.3076142192473403, transmitting_neuron=-6, receiving_neuron=2), 1335: WeightGene(innovationNumber=1335, value=0.2487499187018009, transmitting_neuron=-6, receiving_neuron=3), 1336: WeightGene(innovationNumber=1336, value=0.9390859432236675, transmitting_neuron=-7, receiving_neuron=0), 1337: WeightGene(innovationNumber=1337, value=-1.2546280434416306, transmitting_neuron=-7, receiving_neuron=1), 1338: WeightGene(innovationNumber=1338, value=-0.3321606556095703, transmitting_neuron=-7, receiving_neuron=2), 1339: WeightGene(innovationNumber=1339, value=0.06494794875868395, transmitting_neuron=-7, receiving_neuron=3), 1340: WeightGene(innovationNumber=1340, value=0.04253634236086867, transmitting_neuron=-8, receiving_neuron=0), 1341: WeightGene(innovationNumber=1341, value=-0.3670765927940002, transmitting_neuron=-8, receiving_neuron=1), 1342: WeightGene(innovationNumber=1342, value=1.0098105790141467, transmitting_neuron=-8, receiving_neuron=2), 1343: WeightGene(innovationNumber=1343, value=-1.3122238965843642, transmitting_neuron=-8, receiving_neuron=3)}, bias_chromosme={0: BiasGene(innovationNumber=0, value=1.0362766819744786, parent_neuron=0), 1: BiasGene(innovationNumber=1, value=-0.18985823197314594, parent_neuron=1), 2: BiasGene(innovationNumber=2, value=0.5700734085346386, parent_neuron=2), 3: BiasGene(innovationNumber=3, value=-0.9307184964258022, parent_neuron=3)})
    geneB = Genome(layer_chromosome={0: LayerGene(innovationNumber=0), 1: LayerGene(innovationNumber=1), 2: LayerGene(innovationNumber=2), 3: LayerGene(innovationNumber=3), 4: LayerGene(innovationNumber=4)}, neuron_chromosme={-1: NeruonGene(innovationNumber=-1, parent_layer=0), -2: NeruonGene(innovationNumber=-2, parent_layer=0), -3: NeruonGene(innovationNumber=-3, parent_layer=0), -4: NeruonGene(innovationNumber=-4, parent_layer=0), -5: NeruonGene(innovationNumber=-5, parent_layer=0), -6: NeruonGene(innovationNumber=-6, parent_layer=0), -7: NeruonGene(innovationNumber=-7, parent_layer=0), -8: NeruonGene(innovationNumber=-8, parent_layer=0), 884: NeruonGene(innovationNumber=884, parent_layer=1), 885: NeruonGene(innovationNumber=885, parent_layer=1), 886: NeruonGene(innovationNumber=886, parent_layer=1), 887: NeruonGene(innovationNumber=887, parent_layer=1), 888: NeruonGene(innovationNumber=888, parent_layer=1), 889: NeruonGene(innovationNumber=889, parent_layer=1), 890: NeruonGene(innovationNumber=890, parent_layer=2), 891: NeruonGene(innovationNumber=891, parent_layer=2), 892: NeruonGene(innovationNumber=892, parent_layer=2), 893: NeruonGene(innovationNumber=893, parent_layer=2), 894: NeruonGene(innovationNumber=894, parent_layer=2), 895: NeruonGene(innovationNumber=895, parent_layer=2), 896: NeruonGene(innovationNumber=896, parent_layer=3), 897: NeruonGene(innovationNumber=897, parent_layer=3), 898: NeruonGene(innovationNumber=898, parent_layer=3), 899: NeruonGene(innovationNumber=899, parent_layer=3), 0: NeruonGene(innovationNumber=0, parent_layer=4), 1: NeruonGene(innovationNumber=1, parent_layer=4), 2: NeruonGene(innovationNumber=2, parent_layer=4), 3: NeruonGene(innovationNumber=3, parent_layer=4)}, weight_chromosme={6820: WeightGene(innovationNumber=6820, value=0.386170950319795, transmitting_neuron=-1, receiving_neuron=884), 6821: WeightGene(innovationNumber=6821, value=1.0985023678036936, transmitting_neuron=-1, receiving_neuron=885), 6822: WeightGene(innovationNumber=6822, value=0.19937422879977418, transmitting_neuron=-1, receiving_neuron=886), 6823: WeightGene(innovationNumber=6823, value=-0.28298804218547724, transmitting_neuron=-1, receiving_neuron=887), 6824: WeightGene(innovationNumber=6824, value=-1.200579945520538, transmitting_neuron=-1, receiving_neuron=888), 6825: WeightGene(innovationNumber=6825, value=-1.5131707100258467, transmitting_neuron=-1, receiving_neuron=889), 6826: WeightGene(innovationNumber=6826, value=-0.8730270861319837, transmitting_neuron=-2, receiving_neuron=884), 6827: WeightGene(innovationNumber=6827, value=0.5714497757788116, transmitting_neuron=-2, receiving_neuron=885), 6828: WeightGene(innovationNumber=6828, value=2.206634750441372, transmitting_neuron=-2, receiving_neuron=886), 6829: WeightGene(innovationNumber=6829, value=-0.05038748720310804, transmitting_neuron=-2, receiving_neuron=887), 6830: WeightGene(innovationNumber=6830, value=-0.5084599096304607, transmitting_neuron=-2, receiving_neuron=888), 6831: WeightGene(innovationNumber=6831, value=0.49856519225419704, transmitting_neuron=-2, receiving_neuron=889), 6832: WeightGene(innovationNumber=6832, value=0.6948584418708146, transmitting_neuron=-3, receiving_neuron=884), 6833: WeightGene(innovationNumber=6833, value=0.4961466526745458, transmitting_neuron=-3, receiving_neuron=885), 6834: WeightGene(innovationNumber=6834, value=-0.8813606558733535, transmitting_neuron=-3, receiving_neuron=886), 6835: WeightGene(innovationNumber=6835, value=-1.1111832421531163, transmitting_neuron=-3, receiving_neuron=887), 6836: WeightGene(innovationNumber=6836, value=-1.496216739086193, transmitting_neuron=-3, receiving_neuron=888), 6837: WeightGene(innovationNumber=6837, value=0.5754894965254886, transmitting_neuron=-3, receiving_neuron=889), 6838: WeightGene(innovationNumber=6838, value=1.4741711331154979, transmitting_neuron=-4, receiving_neuron=884), 6839: WeightGene(innovationNumber=6839, value=-0.7988847902307199, transmitting_neuron=-4, receiving_neuron=885), 6840: WeightGene(innovationNumber=6840, value=0.6459037749281739, transmitting_neuron=-4, receiving_neuron=886), 6841: WeightGene(innovationNumber=6841, value=0.5297905365029325, transmitting_neuron=-4, receiving_neuron=887), 6842: WeightGene(innovationNumber=6842, value=-0.850547128949036, transmitting_neuron=-4, receiving_neuron=888), 6843: WeightGene(innovationNumber=6843, value=0.6515030615597821, transmitting_neuron=-4, receiving_neuron=889), 6844: WeightGene(innovationNumber=6844, value=0.7805436815878389, transmitting_neuron=-5, receiving_neuron=884), 6845: WeightGene(innovationNumber=6845, value=-1.04044275059606, transmitting_neuron=-5, receiving_neuron=885), 6846: WeightGene(innovationNumber=6846, value=1.1744269194558083, transmitting_neuron=-5, receiving_neuron=886), 6847: WeightGene(innovationNumber=6847, value=0.9612765254105831, transmitting_neuron=-5, receiving_neuron=887), 6848: WeightGene(innovationNumber=6848, value=2.192800279261608, transmitting_neuron=-5, receiving_neuron=888), 6849: WeightGene(innovationNumber=6849, value=-1.6867540538472696, transmitting_neuron=-5, receiving_neuron=889), 6850: WeightGene(innovationNumber=6850, value=-0.702714287289229, transmitting_neuron=-6, receiving_neuron=884), 6851: WeightGene(innovationNumber=6851, value=-1.0028796931945085, transmitting_neuron=-6, receiving_neuron=885), 6852: WeightGene(innovationNumber=6852, value=0.17307630158340082, transmitting_neuron=-6, receiving_neuron=886), 6853: WeightGene(innovationNumber=6853, value=-0.2742542449393799, transmitting_neuron=-6, receiving_neuron=887), 6854: WeightGene(innovationNumber=6854, value=0.32510966980754386, transmitting_neuron=-6, receiving_neuron=888), 6855: WeightGene(innovationNumber=6855, value=-0.007472867736995181, transmitting_neuron=-6, receiving_neuron=889), 6856: WeightGene(innovationNumber=6856, value=0.024430984494022007, transmitting_neuron=-7, receiving_neuron=884), 6857: WeightGene(innovationNumber=6857, value=0.206154130126838, transmitting_neuron=-7, receiving_neuron=885), 6858: WeightGene(innovationNumber=6858, value=0.4803720751755757, transmitting_neuron=-7, receiving_neuron=886), 6859: WeightGene(innovationNumber=6859, value=-0.42280094381511585, transmitting_neuron=-7, receiving_neuron=887), 6860: WeightGene(innovationNumber=6860, value=1.7044142465452492, transmitting_neuron=-7, receiving_neuron=888), 6861: WeightGene(innovationNumber=6861, value=0.8072527046537816, transmitting_neuron=-7, receiving_neuron=889), 6862: WeightGene(innovationNumber=6862, value=1.2209880622770357, transmitting_neuron=-8, receiving_neuron=884), 6863: WeightGene(innovationNumber=6863, value=0.21486975326536173, transmitting_neuron=-8, receiving_neuron=885), 6864: WeightGene(innovationNumber=6864, value=-0.6958513662770028, transmitting_neuron=-8, receiving_neuron=886), 6865: WeightGene(innovationNumber=6865, value=0.42789729507842433, transmitting_neuron=-8, receiving_neuron=887), 6866: WeightGene(innovationNumber=6866, value=-0.5724904766574975, transmitting_neuron=-8, receiving_neuron=888), 6867: WeightGene(innovationNumber=6867, value=-0.3766813106736633, transmitting_neuron=-8, receiving_neuron=889), 6868: WeightGene(innovationNumber=6868, value=-1.7185119334883465, transmitting_neuron=884, receiving_neuron=890), 6869: WeightGene(innovationNumber=6869, value=0.7697885945145733, transmitting_neuron=884, receiving_neuron=891), 6870: WeightGene(innovationNumber=6870, value=0.32547851994464805, transmitting_neuron=884, receiving_neuron=892), 6871: WeightGene(innovationNumber=6871, value=-0.9868300345176404, transmitting_neuron=884, receiving_neuron=893), 6872: WeightGene(innovationNumber=6872, value=-1.077700491510154, transmitting_neuron=884, receiving_neuron=894), 6873: WeightGene(innovationNumber=6873, value=0.08526084561496905, transmitting_neuron=884, receiving_neuron=895), 6874: WeightGene(innovationNumber=6874, value=0.45810005467607534, transmitting_neuron=885, receiving_neuron=890), 6875: WeightGene(innovationNumber=6875, value=2.171529401910666, transmitting_neuron=885, receiving_neuron=891), 6876: WeightGene(innovationNumber=6876, value=-0.07810424103480107, transmitting_neuron=885, receiving_neuron=892), 6877: WeightGene(innovationNumber=6877, value=-0.4493239974108287, transmitting_neuron=885, receiving_neuron=893), 6878: WeightGene(innovationNumber=6878, value=0.4086948184987608, transmitting_neuron=885, receiving_neuron=894), 6879: WeightGene(innovationNumber=6879, value=-1.127202871875093, transmitting_neuron=885, receiving_neuron=895), 6880: WeightGene(innovationNumber=6880, value=0.04462926107139625, transmitting_neuron=886, receiving_neuron=890), 6881: WeightGene(innovationNumber=6881, value=-1.4000785321124372, transmitting_neuron=886, receiving_neuron=891), 6882: WeightGene(innovationNumber=6882, value=-0.2845994457957653, transmitting_neuron=886, receiving_neuron=892), 6883: WeightGene(innovationNumber=6883, value=-0.9608055046735657, transmitting_neuron=886, receiving_neuron=893), 6884: WeightGene(innovationNumber=6884, value=-0.033224373023862275, transmitting_neuron=886, receiving_neuron=894), 6885: WeightGene(innovationNumber=6885, value=1.2727440217339552, transmitting_neuron=886, receiving_neuron=895), 6886: WeightGene(innovationNumber=6886, value=0.8445230901035783, transmitting_neuron=887, receiving_neuron=890), 6887: WeightGene(innovationNumber=6887, value=-1.2533663374333646, transmitting_neuron=887, receiving_neuron=891), 6888: WeightGene(innovationNumber=6888, value=0.20933537955449794, transmitting_neuron=887, receiving_neuron=892), 6889: WeightGene(innovationNumber=6889, value=-1.1598008805906868, transmitting_neuron=887, receiving_neuron=893), 6890: WeightGene(innovationNumber=6890, value=0.10262053245179009, transmitting_neuron=887, receiving_neuron=894), 6891: WeightGene(innovationNumber=6891, value=-1.3228504827444847, transmitting_neuron=887, receiving_neuron=895), 6892: WeightGene(innovationNumber=6892, value=0.1799803221109847, transmitting_neuron=888, receiving_neuron=890), 6893: WeightGene(innovationNumber=6893, value=-0.1769051074262799, transmitting_neuron=888, receiving_neuron=891), 6894: WeightGene(innovationNumber=6894, value=0.9376883965647588, transmitting_neuron=888, receiving_neuron=892), 6895: WeightGene(innovationNumber=6895, value=-1.1020140482170004, transmitting_neuron=888, receiving_neuron=893), 6896: WeightGene(innovationNumber=6896, value=1.0799519078217688, transmitting_neuron=888, receiving_neuron=894), 6897: WeightGene(innovationNumber=6897, value=0.3848993507199484, transmitting_neuron=888, receiving_neuron=895), 6898: WeightGene(innovationNumber=6898, value=0.6866940633688368, transmitting_neuron=889, receiving_neuron=890), 6899: WeightGene(innovationNumber=6899, value=-0.264881446721834, transmitting_neuron=889, receiving_neuron=891), 6900: WeightGene(innovationNumber=6900, value=-0.36396105205723484, transmitting_neuron=889, receiving_neuron=892), 6901: WeightGene(innovationNumber=6901, value=1.0352738978216993, transmitting_neuron=889, receiving_neuron=893), 6902: WeightGene(innovationNumber=6902, value=-0.758460278765477, transmitting_neuron=889, receiving_neuron=894), 6903: WeightGene(innovationNumber=6903, value=0.0930491341477675, transmitting_neuron=889, receiving_neuron=895), 6904: WeightGene(innovationNumber=6904, value=-1.8718609236250643, transmitting_neuron=890, receiving_neuron=896), 6905: WeightGene(innovationNumber=6905, value=-1.6576165014794033, transmitting_neuron=890, receiving_neuron=897), 6906: WeightGene(innovationNumber=6906, value=1.3187302003769223, transmitting_neuron=890, receiving_neuron=898), 6907: WeightGene(innovationNumber=6907, value=-1.0999613118672749, transmitting_neuron=890, receiving_neuron=899), 6908: WeightGene(innovationNumber=6908, value=-0.45826435223429596, transmitting_neuron=891, receiving_neuron=896), 6909: WeightGene(innovationNumber=6909, value=-0.6124661928462123, transmitting_neuron=891, receiving_neuron=897), 6910: WeightGene(innovationNumber=6910, value=1.899131387037479, transmitting_neuron=891, receiving_neuron=898), 6911: WeightGene(innovationNumber=6911, value=0.7266768845998319, transmitting_neuron=891, receiving_neuron=899), 6912: WeightGene(innovationNumber=6912, value=1.4332024110150527, transmitting_neuron=892, receiving_neuron=896), 6913: WeightGene(innovationNumber=6913, value=0.3793050645263997, transmitting_neuron=892, receiving_neuron=897), 6914: WeightGene(innovationNumber=6914, value=-0.38382793245731567, transmitting_neuron=892, receiving_neuron=898), 6915: WeightGene(innovationNumber=6915, value=-1.322561762185563, transmitting_neuron=892, receiving_neuron=899), 6916: WeightGene(innovationNumber=6916, value=0.1139614392842053, transmitting_neuron=893, receiving_neuron=896), 6917: WeightGene(innovationNumber=6917, value=-0.19892328575250262, transmitting_neuron=893, receiving_neuron=897), 6918: WeightGene(innovationNumber=6918, value=0.5129303401236158, transmitting_neuron=893, receiving_neuron=898), 6919: WeightGene(innovationNumber=6919, value=-0.5845991981306556, transmitting_neuron=893, receiving_neuron=899), 6920: WeightGene(innovationNumber=6920, value=0.2363865251179999, transmitting_neuron=894, receiving_neuron=896), 6921: WeightGene(innovationNumber=6921, value=0.925723494842911, transmitting_neuron=894, receiving_neuron=897), 6922: WeightGene(innovationNumber=6922, value=1.3508836083180409, transmitting_neuron=894, receiving_neuron=898), 6923: WeightGene(innovationNumber=6923, value=-0.27451653771085704, transmitting_neuron=894, receiving_neuron=899), 6924: WeightGene(innovationNumber=6924, value=-0.4124928824512175, transmitting_neuron=895, receiving_neuron=896), 6925: WeightGene(innovationNumber=6925, value=-0.7201804366089667, transmitting_neuron=895, receiving_neuron=897), 6926: WeightGene(innovationNumber=6926, value=-0.5188229730751316, transmitting_neuron=895, receiving_neuron=898), 6927: WeightGene(innovationNumber=6927, value=-1.0686982667266016, transmitting_neuron=895, receiving_neuron=899), 6928: WeightGene(innovationNumber=6928, value=-0.11197849290146916, transmitting_neuron=896, receiving_neuron=0), 6929: WeightGene(innovationNumber=6929, value=-0.5599864465232018, transmitting_neuron=896, receiving_neuron=1), 6930: WeightGene(innovationNumber=6930, value=0.11318364014839918, transmitting_neuron=896, receiving_neuron=2), 6931: WeightGene(innovationNumber=6931, value=-0.28622221400425996, transmitting_neuron=896, receiving_neuron=3), 6932: WeightGene(innovationNumber=6932, value=0.6480506899463544, transmitting_neuron=897, receiving_neuron=0), 6933: WeightGene(innovationNumber=6933, value=-0.7463850986962288, transmitting_neuron=897, receiving_neuron=1), 6934: WeightGene(innovationNumber=6934, value=0.4925491387774756, transmitting_neuron=897, receiving_neuron=2), 6935: WeightGene(innovationNumber=6935, value=-0.3179436778724166, transmitting_neuron=897, receiving_neuron=3), 6936: WeightGene(innovationNumber=6936, value=-0.6208492361049662, transmitting_neuron=898, receiving_neuron=0), 6937: WeightGene(innovationNumber=6937, value=-0.1729212338052114, transmitting_neuron=898, receiving_neuron=1), 6938: WeightGene(innovationNumber=6938, value=0.939399787422675, transmitting_neuron=898, receiving_neuron=2), 6939: WeightGene(innovationNumber=6939, value=-1.4082791359402023, transmitting_neuron=898, receiving_neuron=3), 6940: WeightGene(innovationNumber=6940, value=0.639440950640487, transmitting_neuron=899, receiving_neuron=0), 6941: WeightGene(innovationNumber=6941, value=-0.48862542688837457, transmitting_neuron=899, receiving_neuron=1), 6942: WeightGene(innovationNumber=6942, value=-2.2537991702129845, transmitting_neuron=899, receiving_neuron=2), 6943: WeightGene(innovationNumber=6943, value=-1.6164813388439998, transmitting_neuron=899, receiving_neuron=3)}, bias_chromosme={884: BiasGene(innovationNumber=884, value=0.45009296141416827, parent_neuron=884), 885: BiasGene(innovationNumber=885, value=-0.24273743622897842, parent_neuron=885), 886: BiasGene(innovationNumber=886, value=-0.05766695289206483, parent_neuron=886), 887: BiasGene(innovationNumber=887, value=-0.11551531239057353, parent_neuron=887), 888: BiasGene(innovationNumber=888, value=-0.6331833660087672, parent_neuron=888), 889: BiasGene(innovationNumber=889, value=0.5368916488283422, parent_neuron=889), 890: BiasGene(innovationNumber=890, value=0.2745908109178, parent_neuron=890), 891: BiasGene(innovationNumber=891, value=0.10490156615825141, parent_neuron=891), 892: BiasGene(innovationNumber=892, value=0.2548339814554903, parent_neuron=892), 893: BiasGene(innovationNumber=893, value=1.368569160294943, parent_neuron=893), 894: BiasGene(innovationNumber=894, value=0.146486548873632, parent_neuron=894), 895: BiasGene(innovationNumber=895, value=-0.36023207116104583, parent_neuron=895), 896: BiasGene(innovationNumber=896, value=0.11703775644389053, parent_neuron=896), 897: BiasGene(innovationNumber=897, value=-0.625532742690196, parent_neuron=897), 898: BiasGene(innovationNumber=898, value=-0.050412813837449555, parent_neuron=898), 899: BiasGene(innovationNumber=899, value=-1.6339165950283703, parent_neuron=899), 0: BiasGene(innovationNumber=0, value=1.1773532733005596, parent_neuron=0), 1: BiasGene(innovationNumber=1, value=-0.48771945996711036, parent_neuron=1), 2: BiasGene(innovationNumber=2, value=-0.7477679932497019, parent_neuron=2), 3: BiasGene(innovationNumber=3, value=-0.3058884613749955, parent_neuron=3)})

    for g in range(generations):
        # print("Generation {}".format(g))
        ## Evaualte
        for individual in population:
            evaluate(individual)

        ## Summarize current generation
        best = max(population, key=lambda individual: individual["fitness"])
        average_population_fitness = mean([individual["fitness"] for individual in population])
        print("Best Individual in generation {}: id:{}, fitness {}".format(g, best["id"], best["fitness"]))
        print("Populatation average in generation {} was {}".format(g, average_population_fitness))

        print('------------------------------')

        # Set up population for next generation
        if g != generations - 1:
            population.sort(key=lambda individual: individual["fitness"], reverse=True)

            # Survival selection
            ant_survivors = round(len(population) * survival_rate)
            survivers = population[0:ant_survivors]

            # random_new_rate = 0.00
            # ant_random_new = round(len(population) * random_new_rate)
            ant_random_new = 0

            # Potential_Parent selection
            ant_parents = max(round(parent_rate * pop_size), 2)
            potential_parents = population[0:ant_parents]

            # parent_fitnesses = [ x["fitness"] for x in potential_parents]
            # parent_fitnesses_range = abs ( parent_fitnesses[0] - parent_fitnesses[-1])
            # normalize_fitness_performance = [round(abs(fitness - parent_fitnesses[-1]))/ parent_fitnesses_range for fitness in parent_fitnesses]
            # sumOfP = sum(normalize_fitness_performance)
            # reproduction_probability = [p / sumOfP for p in normalize_fitness_performance]

            ant_children = len(population) - ant_survivors - ant_random_new
            children = []
            for i in range(ant_children):
                # parent = potential_parents[random.randrange(0, len(potential_parents))]
                # Parent selection
                # if parent_selection_weighted_choises:
                #     parent1,parent2 = numpy.random.choice(potential_parents,size=2, p= reproduction_probability,replace=False)
                # else:
                parent1, parent2 = random.sample(potential_parents, k=2)

                # parent1, parent2 = random.choice(potential_parents, 2, weights= reproduction_probability)  # todo weighted selection

                child = crossover(parent1, parent2)
                mutate(child)
                children.append(child)

            population = survivers + children

            # replenish_population_with_new_random(pop_size, population)

            assert len(population) == pop_size

    ### Summary
    best = max(population, key=lambda individual: individual["fitness"])
    print("best Individual got score {}, and looked like:   {}".format(best["fitness"], best))
    average_population_fitness = mean([individual["fitness"] for individual in population])
    print("average population fitness:   {}".format(average_population_fitness))

    env.close()
