
from evaluation import World
from config import Config

from genome import *
import json
import  pandas as pd
import matplotlib.pyplot as plt
from os import listdir, path

from genetic_algorithm import Individ


def graph_fitness_history(filepath, image_folder):
    # filepath = "fitness_history.csv"
    data = pd.read_csv(filepath, delimiter=";", names=["generation", "best_fitness", "average_fitness"])
    print(data)
    fig, ax = plt.subplots()

    ax.plot(data["generation"], data["best_fitness"], label="Best ")
    ax.plot(data["generation"], data["average_fitness"], label="Average")
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.legend()
    plt.show()
    fig.savefig(image_folder+"/fitness_history")


config = Config()
config.ant_simulations = 1
show_graph = True
if show_graph:
    graph_fitness_history(config.saved_fitness_history_file, image_folder= config.playback_folder)

# read from file

world = World(config=config)
# number_of_files = len([name for name in listdir(config.saved_best_individual_per_generation_folder) if path.isfile(path.join(config.saved_best_individual_per_generation_folder, name))])

for g in range(config.generations):
    best_individ_filename = "{}{}".format(config.saved_best_individual_base, g)

    with open(best_individ_filename, 'r') as f:
        dict_data = json.load(f)
        best_individual = eval(dict_data)
        print("Looking at best individual in generation {}".format(g))
        world.evaluate(best_individual, render = True )

world.env.close()