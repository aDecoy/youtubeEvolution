import json
import random
import time
from statistics import mean

from IO_file_functions import write_generation_result_tofile, write_individal_to_file, getIndividSaveFile
from config import Config
from evaluation import World
from genetic_algorithm import crossover, mutate
from genetic_algorithm import initialize_populatin

if __name__ == '__main__':

    config = Config()

    ## Init population
    population = initialize_populatin(config)

    world = World(config=config)

    # geneB =  {'id': 2219, 'genome': Genome(layer_chromosome={0: LayerGene(level=0, innovationNumber=None, activation_function=None), 1: LayerGene(level=1, innovationNumber=None, activation_function=None), 2: LayerGene(level=2, innovationNumber=None, activation_function=None)}, neuron_chromosome={-1: NeuronGene(innovationNumber=-1, parent_layer=0), -2: NeuronGene(innovationNumber=-2, parent_layer=0), -3: NeuronGene(innovationNumber=-3, parent_layer=0), -4: NeuronGene(innovationNumber=-4, parent_layer=0), -5: NeuronGene(innovationNumber=-5, parent_layer=0), -6: NeuronGene(innovationNumber=-6, parent_layer=0), -7: NeuronGene(innovationNumber=-7, parent_layer=0), -8: NeuronGene(innovationNumber=-8, parent_layer=0), 108: NeuronGene(innovationNumber=108, parent_layer=1), 109: NeuronGene(innovationNumber=109, parent_layer=1), 110: NeuronGene(innovationNumber=110, parent_layer=1), 111: NeuronGene(innovationNumber=111, parent_layer=1), 0: NeuronGene(innovationNumber=0, parent_layer=2), 1: NeuronGene(innovationNumber=1, parent_layer=2), 2: NeuronGene(innovationNumber=2, parent_layer=2), 3: NeuronGene(innovationNumber=3, parent_layer=2)}, weight_chromosome={1248: WeightGene(value=0.9433276604105858, innovationNumber=1248, transmitting_neuron=-1, receiving_neuron=108), 1249: WeightGene(value=1.3309991359146838, innovationNumber=1249, transmitting_neuron=-1, receiving_neuron=109), 1250: WeightGene(value=0.8296905620249604, innovationNumber=1250, transmitting_neuron=-1, receiving_neuron=110), 1251: WeightGene(value=0.6173854220434225, innovationNumber=1251, transmitting_neuron=-1, receiving_neuron=111), 1252: WeightGene(value=1.2315522142001747, innovationNumber=1252, transmitting_neuron=-2, receiving_neuron=108), 1253: WeightGene(value=0.1717471046074385, innovationNumber=1253, transmitting_neuron=-2, receiving_neuron=109), 1254: WeightGene(value=-0.8730469043090361, innovationNumber=1254, transmitting_neuron=-2, receiving_neuron=110), 1255: WeightGene(value=0.14062688160851422, innovationNumber=1255, transmitting_neuron=-2, receiving_neuron=111), 1256: WeightGene(value=0.09762408450695048, innovationNumber=1256, transmitting_neuron=-3, receiving_neuron=108), 1257: WeightGene(value=0.19577176253156636, innovationNumber=1257, transmitting_neuron=-3, receiving_neuron=109), 1258: WeightGene(value=-0.15640976446709376, innovationNumber=1258, transmitting_neuron=-3, receiving_neuron=110), 1259: WeightGene(value=-0.0463296916910145, innovationNumber=1259, transmitting_neuron=-3, receiving_neuron=111), 1260: WeightGene(value=0.4322197991572294, innovationNumber=1260, transmitting_neuron=-4, receiving_neuron=108), 1261: WeightGene(value=-0.4405215085801435, innovationNumber=1261, transmitting_neuron=-4, receiving_neuron=109), 1262: WeightGene(value=-0.6708439704209194, innovationNumber=1262, transmitting_neuron=-4, receiving_neuron=110), 1263: WeightGene(value=1.2137582682183132, innovationNumber=1263, transmitting_neuron=-4, receiving_neuron=111), 1264: WeightGene(value=-1.2051739653304212, innovationNumber=1264, transmitting_neuron=-5, receiving_neuron=108), 1265: WeightGene(value=-0.38875891149916797, innovationNumber=1265, transmitting_neuron=-5, receiving_neuron=109), 1266: WeightGene(value=-0.006313907304450916, innovationNumber=1266, transmitting_neuron=-5, receiving_neuron=110), 1267: WeightGene(value=-0.5561866113517258, innovationNumber=1267, transmitting_neuron=-5, receiving_neuron=111), 1268: WeightGene(value=0.3031599686327767, innovationNumber=1268, transmitting_neuron=-6, receiving_neuron=108), 1269: WeightGene(value=-0.8095897889960313, innovationNumber=1269, transmitting_neuron=-6, receiving_neuron=109), 1270: WeightGene(value=0.1586314698285315, innovationNumber=1270, transmitting_neuron=-6, receiving_neuron=110), 1271: WeightGene(value=0.9918726184859533, innovationNumber=1271, transmitting_neuron=-6, receiving_neuron=111), 1272: WeightGene(value=-0.7355105369662676, innovationNumber=1272, transmitting_neuron=-7, receiving_neuron=108), 1273: WeightGene(value=-0.3899354078985431, innovationNumber=1273, transmitting_neuron=-7, receiving_neuron=109), 1274: WeightGene(value=-0.4394729613030534, innovationNumber=1274, transmitting_neuron=-7, receiving_neuron=110), 1275: WeightGene(value=0.22655646691765957, innovationNumber=1275, transmitting_neuron=-7, receiving_neuron=111), 1276: WeightGene(value=0.8268646338432031, innovationNumber=1276, transmitting_neuron=-8, receiving_neuron=108), 1277: WeightGene(value=-0.848737706344886, innovationNumber=1277, transmitting_neuron=-8, receiving_neuron=109), 1278: WeightGene(value=-0.9770976777158128, innovationNumber=1278, transmitting_neuron=-8, receiving_neuron=110), 1279: WeightGene(value=0.30049549592053476, innovationNumber=1279, transmitting_neuron=-8, receiving_neuron=111), 1280: WeightGene(value=0.7395496008902067, innovationNumber=1280, transmitting_neuron=108, receiving_neuron=0), 1281: WeightGene(value=-0.6690782608690221, innovationNumber=1281, transmitting_neuron=108, receiving_neuron=1), 1282: WeightGene(value=-0.08775073763817653, innovationNumber=1282, transmitting_neuron=108, receiving_neuron=2), 1283: WeightGene(value=0.14110829925703278, innovationNumber=1283, transmitting_neuron=108, receiving_neuron=3), 1284: WeightGene(value=0.6927498596772106, innovationNumber=1284, transmitting_neuron=109, receiving_neuron=0), 1285: WeightGene(value=0.17024718669516548, innovationNumber=1285, transmitting_neuron=109, receiving_neuron=1), 1286: WeightGene(value=0.7420756748955472, innovationNumber=1286, transmitting_neuron=109, receiving_neuron=2), 1287: WeightGene(value=0.3721029843758649, innovationNumber=1287, transmitting_neuron=109, receiving_neuron=3), 1288: WeightGene(value=-1.1755119055162369, innovationNumber=1288, transmitting_neuron=110, receiving_neuron=0), 1289: WeightGene(value=0.16905377466895763, innovationNumber=1289, transmitting_neuron=110, receiving_neuron=1), 1290: WeightGene(value=0.012035969476737685, innovationNumber=1290, transmitting_neuron=110, receiving_neuron=2), 1291: WeightGene(value=-0.4754311329467581, innovationNumber=1291, transmitting_neuron=110, receiving_neuron=3), 1292: WeightGene(value=1.1958463640363421, innovationNumber=1292, transmitting_neuron=111, receiving_neuron=0), 1293: WeightGene(value=0.043463871930397635, innovationNumber=1293, transmitting_neuron=111, receiving_neuron=1), 1294: WeightGene(value=0.026669355679097284, innovationNumber=1294, transmitting_neuron=111, receiving_neuron=2), 1295: WeightGene(value=-0.03835137731020505, innovationNumber=1295, transmitting_neuron=111, receiving_neuron=3)}, bias_chromosome={108: BiasGene(value=0.7668089188110419, parent_neuron=108, innovationNumber=None), 109: BiasGene(value=0.6232083939584674, parent_neuron=109, innovationNumber=None), 110: BiasGene(value=0.1525173479877075, parent_neuron=110, innovationNumber=None), 111: BiasGene(value=0.38175093124723825, parent_neuron=111, innovationNumber=None), 0: BiasGene(value=0.46697282789873307, parent_neuron=0, innovationNumber=None), 1: BiasGene(value=0.49660603710994705, parent_neuron=1, innovationNumber=None), 2: BiasGene(value=0.6718369588691699, parent_neuron=2, innovationNumber=None), 3: BiasGene(value=0.736153780701598, parent_neuron=3, innovationNumber=None)}), 'fitness': -8.393776440787112}
    # layer_sizes = [ant_input_neruons, 4, ant_output_neruons]

    for g in range(config.generations):
        # print("Generation {}".format(g))
        ## Evaualte
        for individual in population:
            world.evaluate(individual, ant_simulations=config.ant_simulations)

        after_evaluation = time.time()
        ## Summarize current generation
        best = max(population, key=lambda individual: individual.fitness)
        average_population_fitness = mean([individual.fitness for individual in population])
        print("Best Individual in generation {}: id:{}, fitness {}".format(g, best.id, best.fitness))
        print("Populatation average in generation {} was {}".format(g, average_population_fitness))

        print('------------------------------')
        write_generation_result_tofile(filename=config.save_score_file,
                                       best_fitness=best.fitness, average_fitness=average_population_fitness,
                                       generation=g
                                       )
        write_individal_to_file(save_best_individual_base=config.save_best_individual_base ,individ=best, generation=g)

        # Set up population for next generation
        if g != config.generations - 1:
            population.sort(key=lambda individual: individual.fitness, reverse=True)

            # Survival selection
            survival_rate = 0.10
            ant_survivors = round(len(population) * survival_rate)
            survivers = population[0:ant_survivors]

            # random_new_rate = 0.00
            # ant_random_new = round(len(population) * random_new_rate)
            ant_random_new = 0

            # Potential_Parent selection
            parent_rate = 0.10
            ant_parents = max(round(parent_rate * config.pop_size), 2)
            potential_parents = population[0:ant_parents]

            ant_children = len(population) - ant_survivors - ant_random_new
            children = []
            for i in range(ant_children):
                # parent = potential_parents[random.randrange(0, len(potential_parents))]
                # Parent selection
                parent1, parent2 = random.sample(potential_parents, k=2)

                child = crossover(parent1, parent2,config=config)
                mutate(child,config=config)
                children.append(child)

            population = survivers + children

            # replenish_population_with_new_random(pop_size, population)

            assert len(population) == config.pop_size

            non_evalutation_time = time.time() - after_evaluation
            print(non_evalutation_time)

    ### Summary
    best = max(population, key=lambda individual: individual.fitness)
    print("best Individual got score {}, and looked like:   {}".format(best.fitness, best))
    average_population_fitness = mean([individual.fitness for individual in population])
    print("average population fitness:   {}".format(average_population_fitness))

    ## playback

    for g in range(config.generations):
        best_individual_file = getIndividSaveFile(config.save_best_individual_base, g)
        print("Best in generation {}".format(g))
        with open(best_individual_file, 'r') as f:
            dict_data = json.load(f)
            best_individual = eval(dict_data)

            world.evaluate(best_individual, ant_simulations=3, render=False)

    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv(config.save_score_file, sep=";", names=["generation", "best fitness", "avg_fitness"])
    fig, ax = plt.subplots()
    ax.plot(data["generation"], data["best fitness"])
    # data.plot(x="generation", y="best fitness")
    plt.show()
    fig.savefig(config.save_score_file_folder + "/test")
    world.env.close()
