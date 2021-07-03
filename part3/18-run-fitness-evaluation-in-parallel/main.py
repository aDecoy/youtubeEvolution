import csv
import json
from statistics import mean

from evaluation import World, evaluate
from genetic_algorithm import initialize_population, crossover, mutate
from genome import *
from shutil import rmtree
import time

def write_generation_ressult_to_file(filename : str, best_fitness:float, average_fitness: float, generation:int):
    with open(filename, 'a+') as f:
        # csv_row =  {"generation":generation, "best_fitness": best_fitness, "average_fitness": average_fitness}
        csv_row =  [generation,  best_fitness, average_fitness]
        writer_object = csv.writer(f,delimiter=";")
        writer_object.writerow(csv_row)


def write_best_individual_to_file(individual: dict, generation: Config, config: Config):
    filename = config.saved_best_individual_base+str(generation)
    with open(filename, 'w') as f:
        json.dump(str(individual), f)


def first_N_items(dictionary : dict, n :int):
    counter : int = 0
    new_dict: dict = {}
    for key, value in dictionary.items():
        if counter >= n:
            break
        new_dict[key] = value
        counter +=1

    return new_dict

if __name__ == '__main__':
    see_time = True
    t0 = time.time()

    config = Config()
    rmtree(config.playback_folder)
    config = Config()

    world = World(config=config)

    ## Init population
    population = initialize_population(config)

    t1 = time.time()
    if see_time :
        print("time to start up before starting on the generations : {}s".format(t1-t0))

    geneB = Genome(layer_chromosome={0: LayerGene(innovationNumber=0), 1: LayerGene(innovationNumber=1), 2: LayerGene(innovationNumber=2), 3: LayerGene(innovationNumber=3), 4: LayerGene(innovationNumber=4)}, neuron_chromosme={-1: NeruonGene(innovationNumber=-1, parent_layer=0), -2: NeruonGene(innovationNumber=-2, parent_layer=0), -3: NeruonGene(innovationNumber=-3, parent_layer=0), -4: NeruonGene(innovationNumber=-4, parent_layer=0), -5: NeruonGene(innovationNumber=-5, parent_layer=0), -6: NeruonGene(innovationNumber=-6, parent_layer=0), -7: NeruonGene(innovationNumber=-7, parent_layer=0), -8: NeruonGene(innovationNumber=-8, parent_layer=0), 884: NeruonGene(innovationNumber=884, parent_layer=1), 885: NeruonGene(innovationNumber=885, parent_layer=1), 886: NeruonGene(innovationNumber=886, parent_layer=1), 887: NeruonGene(innovationNumber=887, parent_layer=1), 888: NeruonGene(innovationNumber=888, parent_layer=1), 889: NeruonGene(innovationNumber=889, parent_layer=1), 890: NeruonGene(innovationNumber=890, parent_layer=2), 891: NeruonGene(innovationNumber=891, parent_layer=2), 892: NeruonGene(innovationNumber=892, parent_layer=2), 893: NeruonGene(innovationNumber=893, parent_layer=2), 894: NeruonGene(innovationNumber=894, parent_layer=2), 895: NeruonGene(innovationNumber=895, parent_layer=2), 896: NeruonGene(innovationNumber=896, parent_layer=3), 897: NeruonGene(innovationNumber=897, parent_layer=3), 898: NeruonGene(innovationNumber=898, parent_layer=3), 899: NeruonGene(innovationNumber=899, parent_layer=3), 0: NeruonGene(innovationNumber=0, parent_layer=4), 1: NeruonGene(innovationNumber=1, parent_layer=4), 2: NeruonGene(innovationNumber=2, parent_layer=4), 3: NeruonGene(innovationNumber=3, parent_layer=4)}, weight_chromosme={6820: WeightGene(innovationNumber=6820, value=0.386170950319795, transmitting_neuron=-1, receiving_neuron=884), 6821: WeightGene(innovationNumber=6821, value=1.0985023678036936, transmitting_neuron=-1, receiving_neuron=885), 6822: WeightGene(innovationNumber=6822, value=0.19937422879977418, transmitting_neuron=-1, receiving_neuron=886), 6823: WeightGene(innovationNumber=6823, value=-0.28298804218547724, transmitting_neuron=-1, receiving_neuron=887), 6824: WeightGene(innovationNumber=6824, value=-1.200579945520538, transmitting_neuron=-1, receiving_neuron=888), 6825: WeightGene(innovationNumber=6825, value=-1.5131707100258467, transmitting_neuron=-1, receiving_neuron=889), 6826: WeightGene(innovationNumber=6826, value=-0.8730270861319837, transmitting_neuron=-2, receiving_neuron=884), 6827: WeightGene(innovationNumber=6827, value=0.5714497757788116, transmitting_neuron=-2, receiving_neuron=885), 6828: WeightGene(innovationNumber=6828, value=2.206634750441372, transmitting_neuron=-2, receiving_neuron=886), 6829: WeightGene(innovationNumber=6829, value=-0.05038748720310804, transmitting_neuron=-2, receiving_neuron=887), 6830: WeightGene(innovationNumber=6830, value=-0.5084599096304607, transmitting_neuron=-2, receiving_neuron=888), 6831: WeightGene(innovationNumber=6831, value=0.49856519225419704, transmitting_neuron=-2, receiving_neuron=889), 6832: WeightGene(innovationNumber=6832, value=0.6948584418708146, transmitting_neuron=-3, receiving_neuron=884), 6833: WeightGene(innovationNumber=6833, value=0.4961466526745458, transmitting_neuron=-3, receiving_neuron=885), 6834: WeightGene(innovationNumber=6834, value=-0.8813606558733535, transmitting_neuron=-3, receiving_neuron=886), 6835: WeightGene(innovationNumber=6835, value=-1.1111832421531163, transmitting_neuron=-3, receiving_neuron=887), 6836: WeightGene(innovationNumber=6836, value=-1.496216739086193, transmitting_neuron=-3, receiving_neuron=888), 6837: WeightGene(innovationNumber=6837, value=0.5754894965254886, transmitting_neuron=-3, receiving_neuron=889), 6838: WeightGene(innovationNumber=6838, value=1.4741711331154979, transmitting_neuron=-4, receiving_neuron=884), 6839: WeightGene(innovationNumber=6839, value=-0.7988847902307199, transmitting_neuron=-4, receiving_neuron=885), 6840: WeightGene(innovationNumber=6840, value=0.6459037749281739, transmitting_neuron=-4, receiving_neuron=886), 6841: WeightGene(innovationNumber=6841, value=0.5297905365029325, transmitting_neuron=-4, receiving_neuron=887), 6842: WeightGene(innovationNumber=6842, value=-0.850547128949036, transmitting_neuron=-4, receiving_neuron=888), 6843: WeightGene(innovationNumber=6843, value=0.6515030615597821, transmitting_neuron=-4, receiving_neuron=889), 6844: WeightGene(innovationNumber=6844, value=0.7805436815878389, transmitting_neuron=-5, receiving_neuron=884), 6845: WeightGene(innovationNumber=6845, value=-1.04044275059606, transmitting_neuron=-5, receiving_neuron=885), 6846: WeightGene(innovationNumber=6846, value=1.1744269194558083, transmitting_neuron=-5, receiving_neuron=886), 6847: WeightGene(innovationNumber=6847, value=0.9612765254105831, transmitting_neuron=-5, receiving_neuron=887), 6848: WeightGene(innovationNumber=6848, value=2.192800279261608, transmitting_neuron=-5, receiving_neuron=888), 6849: WeightGene(innovationNumber=6849, value=-1.6867540538472696, transmitting_neuron=-5, receiving_neuron=889), 6850: WeightGene(innovationNumber=6850, value=-0.702714287289229, transmitting_neuron=-6, receiving_neuron=884), 6851: WeightGene(innovationNumber=6851, value=-1.0028796931945085, transmitting_neuron=-6, receiving_neuron=885), 6852: WeightGene(innovationNumber=6852, value=0.17307630158340082, transmitting_neuron=-6, receiving_neuron=886), 6853: WeightGene(innovationNumber=6853, value=-0.2742542449393799, transmitting_neuron=-6, receiving_neuron=887), 6854: WeightGene(innovationNumber=6854, value=0.32510966980754386, transmitting_neuron=-6, receiving_neuron=888), 6855: WeightGene(innovationNumber=6855, value=-0.007472867736995181, transmitting_neuron=-6, receiving_neuron=889), 6856: WeightGene(innovationNumber=6856, value=0.024430984494022007, transmitting_neuron=-7, receiving_neuron=884), 6857: WeightGene(innovationNumber=6857, value=0.206154130126838, transmitting_neuron=-7, receiving_neuron=885), 6858: WeightGene(innovationNumber=6858, value=0.4803720751755757, transmitting_neuron=-7, receiving_neuron=886), 6859: WeightGene(innovationNumber=6859, value=-0.42280094381511585, transmitting_neuron=-7, receiving_neuron=887), 6860: WeightGene(innovationNumber=6860, value=1.7044142465452492, transmitting_neuron=-7, receiving_neuron=888), 6861: WeightGene(innovationNumber=6861, value=0.8072527046537816, transmitting_neuron=-7, receiving_neuron=889), 6862: WeightGene(innovationNumber=6862, value=1.2209880622770357, transmitting_neuron=-8, receiving_neuron=884), 6863: WeightGene(innovationNumber=6863, value=0.21486975326536173, transmitting_neuron=-8, receiving_neuron=885), 6864: WeightGene(innovationNumber=6864, value=-0.6958513662770028, transmitting_neuron=-8, receiving_neuron=886), 6865: WeightGene(innovationNumber=6865, value=0.42789729507842433, transmitting_neuron=-8, receiving_neuron=887), 6866: WeightGene(innovationNumber=6866, value=-0.5724904766574975, transmitting_neuron=-8, receiving_neuron=888), 6867: WeightGene(innovationNumber=6867, value=-0.3766813106736633, transmitting_neuron=-8, receiving_neuron=889), 6868: WeightGene(innovationNumber=6868, value=-1.7185119334883465, transmitting_neuron=884, receiving_neuron=890), 6869: WeightGene(innovationNumber=6869, value=0.7697885945145733, transmitting_neuron=884, receiving_neuron=891), 6870: WeightGene(innovationNumber=6870, value=0.32547851994464805, transmitting_neuron=884, receiving_neuron=892), 6871: WeightGene(innovationNumber=6871, value=-0.9868300345176404, transmitting_neuron=884, receiving_neuron=893), 6872: WeightGene(innovationNumber=6872, value=-1.077700491510154, transmitting_neuron=884, receiving_neuron=894), 6873: WeightGene(innovationNumber=6873, value=0.08526084561496905, transmitting_neuron=884, receiving_neuron=895), 6874: WeightGene(innovationNumber=6874, value=0.45810005467607534, transmitting_neuron=885, receiving_neuron=890), 6875: WeightGene(innovationNumber=6875, value=2.171529401910666, transmitting_neuron=885, receiving_neuron=891), 6876: WeightGene(innovationNumber=6876, value=-0.07810424103480107, transmitting_neuron=885, receiving_neuron=892), 6877: WeightGene(innovationNumber=6877, value=-0.4493239974108287, transmitting_neuron=885, receiving_neuron=893), 6878: WeightGene(innovationNumber=6878, value=0.4086948184987608, transmitting_neuron=885, receiving_neuron=894), 6879: WeightGene(innovationNumber=6879, value=-1.127202871875093, transmitting_neuron=885, receiving_neuron=895), 6880: WeightGene(innovationNumber=6880, value=0.04462926107139625, transmitting_neuron=886, receiving_neuron=890), 6881: WeightGene(innovationNumber=6881, value=-1.4000785321124372, transmitting_neuron=886, receiving_neuron=891), 6882: WeightGene(innovationNumber=6882, value=-0.2845994457957653, transmitting_neuron=886, receiving_neuron=892), 6883: WeightGene(innovationNumber=6883, value=-0.9608055046735657, transmitting_neuron=886, receiving_neuron=893), 6884: WeightGene(innovationNumber=6884, value=-0.033224373023862275, transmitting_neuron=886, receiving_neuron=894), 6885: WeightGene(innovationNumber=6885, value=1.2727440217339552, transmitting_neuron=886, receiving_neuron=895), 6886: WeightGene(innovationNumber=6886, value=0.8445230901035783, transmitting_neuron=887, receiving_neuron=890), 6887: WeightGene(innovationNumber=6887, value=-1.2533663374333646, transmitting_neuron=887, receiving_neuron=891), 6888: WeightGene(innovationNumber=6888, value=0.20933537955449794, transmitting_neuron=887, receiving_neuron=892), 6889: WeightGene(innovationNumber=6889, value=-1.1598008805906868, transmitting_neuron=887, receiving_neuron=893), 6890: WeightGene(innovationNumber=6890, value=0.10262053245179009, transmitting_neuron=887, receiving_neuron=894), 6891: WeightGene(innovationNumber=6891, value=-1.3228504827444847, transmitting_neuron=887, receiving_neuron=895), 6892: WeightGene(innovationNumber=6892, value=0.1799803221109847, transmitting_neuron=888, receiving_neuron=890), 6893: WeightGene(innovationNumber=6893, value=-0.1769051074262799, transmitting_neuron=888, receiving_neuron=891), 6894: WeightGene(innovationNumber=6894, value=0.9376883965647588, transmitting_neuron=888, receiving_neuron=892), 6895: WeightGene(innovationNumber=6895, value=-1.1020140482170004, transmitting_neuron=888, receiving_neuron=893), 6896: WeightGene(innovationNumber=6896, value=1.0799519078217688, transmitting_neuron=888, receiving_neuron=894), 6897: WeightGene(innovationNumber=6897, value=0.3848993507199484, transmitting_neuron=888, receiving_neuron=895), 6898: WeightGene(innovationNumber=6898, value=0.6866940633688368, transmitting_neuron=889, receiving_neuron=890), 6899: WeightGene(innovationNumber=6899, value=-0.264881446721834, transmitting_neuron=889, receiving_neuron=891), 6900: WeightGene(innovationNumber=6900, value=-0.36396105205723484, transmitting_neuron=889, receiving_neuron=892), 6901: WeightGene(innovationNumber=6901, value=1.0352738978216993, transmitting_neuron=889, receiving_neuron=893), 6902: WeightGene(innovationNumber=6902, value=-0.758460278765477, transmitting_neuron=889, receiving_neuron=894), 6903: WeightGene(innovationNumber=6903, value=0.0930491341477675, transmitting_neuron=889, receiving_neuron=895), 6904: WeightGene(innovationNumber=6904, value=-1.8718609236250643, transmitting_neuron=890, receiving_neuron=896), 6905: WeightGene(innovationNumber=6905, value=-1.6576165014794033, transmitting_neuron=890, receiving_neuron=897), 6906: WeightGene(innovationNumber=6906, value=1.3187302003769223, transmitting_neuron=890, receiving_neuron=898), 6907: WeightGene(innovationNumber=6907, value=-1.0999613118672749, transmitting_neuron=890, receiving_neuron=899), 6908: WeightGene(innovationNumber=6908, value=-0.45826435223429596, transmitting_neuron=891, receiving_neuron=896), 6909: WeightGene(innovationNumber=6909, value=-0.6124661928462123, transmitting_neuron=891, receiving_neuron=897), 6910: WeightGene(innovationNumber=6910, value=1.899131387037479, transmitting_neuron=891, receiving_neuron=898), 6911: WeightGene(innovationNumber=6911, value=0.7266768845998319, transmitting_neuron=891, receiving_neuron=899), 6912: WeightGene(innovationNumber=6912, value=1.4332024110150527, transmitting_neuron=892, receiving_neuron=896), 6913: WeightGene(innovationNumber=6913, value=0.3793050645263997, transmitting_neuron=892, receiving_neuron=897), 6914: WeightGene(innovationNumber=6914, value=-0.38382793245731567, transmitting_neuron=892, receiving_neuron=898), 6915: WeightGene(innovationNumber=6915, value=-1.322561762185563, transmitting_neuron=892, receiving_neuron=899), 6916: WeightGene(innovationNumber=6916, value=0.1139614392842053, transmitting_neuron=893, receiving_neuron=896), 6917: WeightGene(innovationNumber=6917, value=-0.19892328575250262, transmitting_neuron=893, receiving_neuron=897), 6918: WeightGene(innovationNumber=6918, value=0.5129303401236158, transmitting_neuron=893, receiving_neuron=898), 6919: WeightGene(innovationNumber=6919, value=-0.5845991981306556, transmitting_neuron=893, receiving_neuron=899), 6920: WeightGene(innovationNumber=6920, value=0.2363865251179999, transmitting_neuron=894, receiving_neuron=896), 6921: WeightGene(innovationNumber=6921, value=0.925723494842911, transmitting_neuron=894, receiving_neuron=897), 6922: WeightGene(innovationNumber=6922, value=1.3508836083180409, transmitting_neuron=894, receiving_neuron=898), 6923: WeightGene(innovationNumber=6923, value=-0.27451653771085704, transmitting_neuron=894, receiving_neuron=899), 6924: WeightGene(innovationNumber=6924, value=-0.4124928824512175, transmitting_neuron=895, receiving_neuron=896), 6925: WeightGene(innovationNumber=6925, value=-0.7201804366089667, transmitting_neuron=895, receiving_neuron=897), 6926: WeightGene(innovationNumber=6926, value=-0.5188229730751316, transmitting_neuron=895, receiving_neuron=898), 6927: WeightGene(innovationNumber=6927, value=-1.0686982667266016, transmitting_neuron=895, receiving_neuron=899), 6928: WeightGene(innovationNumber=6928, value=-0.11197849290146916, transmitting_neuron=896, receiving_neuron=0), 6929: WeightGene(innovationNumber=6929, value=-0.5599864465232018, transmitting_neuron=896, receiving_neuron=1), 6930: WeightGene(innovationNumber=6930, value=0.11318364014839918, transmitting_neuron=896, receiving_neuron=2), 6931: WeightGene(innovationNumber=6931, value=-0.28622221400425996, transmitting_neuron=896, receiving_neuron=3), 6932: WeightGene(innovationNumber=6932, value=0.6480506899463544, transmitting_neuron=897, receiving_neuron=0), 6933: WeightGene(innovationNumber=6933, value=-0.7463850986962288, transmitting_neuron=897, receiving_neuron=1), 6934: WeightGene(innovationNumber=6934, value=0.4925491387774756, transmitting_neuron=897, receiving_neuron=2), 6935: WeightGene(innovationNumber=6935, value=-0.3179436778724166, transmitting_neuron=897, receiving_neuron=3), 6936: WeightGene(innovationNumber=6936, value=-0.6208492361049662, transmitting_neuron=898, receiving_neuron=0), 6937: WeightGene(innovationNumber=6937, value=-0.1729212338052114, transmitting_neuron=898, receiving_neuron=1), 6938: WeightGene(innovationNumber=6938, value=0.939399787422675, transmitting_neuron=898, receiving_neuron=2), 6939: WeightGene(innovationNumber=6939, value=-1.4082791359402023, transmitting_neuron=898, receiving_neuron=3), 6940: WeightGene(innovationNumber=6940, value=0.639440950640487, transmitting_neuron=899, receiving_neuron=0), 6941: WeightGene(innovationNumber=6941, value=-0.48862542688837457, transmitting_neuron=899, receiving_neuron=1), 6942: WeightGene(innovationNumber=6942, value=-2.2537991702129845, transmitting_neuron=899, receiving_neuron=2), 6943: WeightGene(innovationNumber=6943, value=-1.6164813388439998, transmitting_neuron=899, receiving_neuron=3)}, bias_chromosme={884: BiasGene(innovationNumber=884, value=0.45009296141416827, parent_neuron=884), 885: BiasGene(innovationNumber=885, value=-0.24273743622897842, parent_neuron=885), 886: BiasGene(innovationNumber=886, value=-0.05766695289206483, parent_neuron=886), 887: BiasGene(innovationNumber=887, value=-0.11551531239057353, parent_neuron=887), 888: BiasGene(innovationNumber=888, value=-0.6331833660087672, parent_neuron=888), 889: BiasGene(innovationNumber=889, value=0.5368916488283422, parent_neuron=889), 890: BiasGene(innovationNumber=890, value=0.2745908109178, parent_neuron=890), 891: BiasGene(innovationNumber=891, value=0.10490156615825141, parent_neuron=891), 892: BiasGene(innovationNumber=892, value=0.2548339814554903, parent_neuron=892), 893: BiasGene(innovationNumber=893, value=1.368569160294943, parent_neuron=893), 894: BiasGene(innovationNumber=894, value=0.146486548873632, parent_neuron=894), 895: BiasGene(innovationNumber=895, value=-0.36023207116104583, parent_neuron=895), 896: BiasGene(innovationNumber=896, value=0.11703775644389053, parent_neuron=896), 897: BiasGene(innovationNumber=897, value=-0.625532742690196, parent_neuron=897), 898: BiasGene(innovationNumber=898, value=-0.050412813837449555, parent_neuron=898), 899: BiasGene(innovationNumber=899, value=-1.6339165950283703, parent_neuron=899), 0: BiasGene(innovationNumber=0, value=1.1773532733005596, parent_neuron=0), 1: BiasGene(innovationNumber=1, value=-0.48771945996711036, parent_neuron=1), 2: BiasGene(innovationNumber=2, value=-0.7477679932497019, parent_neuron=2), 3: BiasGene(innovationNumber=3, value=-0.3058884613749955, parent_neuron=3)})

    for g in range(config.generations):
        # print("Generation {}".format(g))
        ## Evaualte
        t3 = time.time()
        population = evaluate(population=population.values(), config=config)
        t4 = time.time()
        # if see_time:
        print("Time spent on evaluating the population {}s".format(t4-t3))
        # for individual in population.values():
        #     world.evaluate(individual)

        ## Summarize current generation
        best = max(population.values(), key=lambda individual: individual.fitness)
        average_population_fitness = mean([individual.fitness for individual in population.values()])
        print("Best Individual in generation {}: id:{}, fitness {}".format(g, best.id, best.fitness))
        print("Populatation average in generation {} was {}".format(g, average_population_fitness))

        write_generation_ressult_to_file(filename=config.saved_fitness_history_file,generation=g,best_fitness=best.fitness,average_fitness=average_population_fitness )
        write_best_individual_to_file(individual=best, generation=g, config=config)
        print('------------------------------')

        # Set up population for next generation
        if g != config.generations - 1:
            population = dict(sorted(population.items(), key=lambda key_value_tuple: key_value_tuple[1].fitness, reverse=True))
            # Survival selection
            ant_survivors = round(len(population) * config.survival_rate)
            # survivers = population[0:ant_survivors]

            survivers = first_N_items(population,ant_survivors)

            # Potential_Parent selection
            ant_parents = max(round(config.parent_rate * config.pop_size), 2)
            # potential_parents = population[0:ant_parents]
            potential_parents = first_N_items(population,ant_parents)

            ant_children = len(population) - ant_survivors
            children = {}
            for i in range(ant_children):
                # parent = potential_parents[random.randrange(0, len(potential_parents))]
                # Parent selection
                parent1, parent2 = random.sample(list(potential_parents.values()), k=2)
                child = crossover(parent1, parent2,config=config)
                mutate(child,config=config)
                children[child.id]=child

            population = survivers | children
            assert len(population) == config.pop_size

            if see_time:
                t5 = time.time()
                print("Time spent on not evaluating individuals {}".format(t5-t4))

    ### Summary
    best = max(population.values(), key=lambda individual: individual.fitness)
    print("best Individual got score {}, and looked like:   {}".format(best.fitness, best))
    average_population_fitness = mean([individual.fitness for individual in population.values()])
    print("average population fitness:   {}".format(average_population_fitness))

    world.env.close()

    filename = config.playback_folder + "/best_individual"
    with open(filename, 'w') as f:
        json.dump(best.__str__(),f)

