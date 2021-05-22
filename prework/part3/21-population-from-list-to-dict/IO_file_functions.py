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


def write_generation_result_tofile(filename: str,
        best_fitness: float, average_fitness: float, generation: int, delimiter: str = ";"
):
    with open(filename, 'a+') as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow([generation, round(best_fitness, 4), round(average_fitness, 4)])



def write_individal_to_file(save_best_individual_base ,individ: dict, generation: int, delimiter=";"):
    filename = getIndividSaveFile(save_best_individual_base, generation)
    with open(filename, 'w') as f:
        json.dump(individ.__str__(), f)


def getIndividSaveFile(save_best_individual_base, generation):
    return "{}{}.json".format(save_best_individual_base, generation)
