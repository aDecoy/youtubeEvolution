import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from IO_file_functions import getIndividSaveFile
import json
from evaluation import World
from genome import *

import json

from config import Config
from evaluation import World

## playback

from genetic_algorithm import Individ


config = Config
world = World(config=config)

for g in range(config.generations):
    best_individual_file = getIndividSaveFile(config.save_best_individual_base, g)
    print("Best in generation {}".format(g))
    with open(best_individual_file, 'r') as f:
        dict_data = json.load(f)
        best_individual = eval(dict_data)

        world.evaluate(best_individual, ant_simulations=1,render=True)


data = pd.read_csv(config.save_score_file, sep=";", names=["generation", "best fitness", "avg_fitness"])
fig, ax = plt.subplots()
ax.plot(data["generation"], data["best fitness"])
# data.plot(x="generation", y="best fitness")
plt.show()
fig.savefig(config.save_score_file_folder + "/test")
world.env.close()
