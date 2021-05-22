from dataclasses import dataclass, field
from itertools import count
from config import Config
from simulation_stuff import Individ


@dataclass
class Specie:
    key: int
    created_on: int
    last_imporved: int
    repesentaive: Individ
    members: dict
    fitness: float
    adjusted_fitness: float
    fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitness(self):
        return [m.fitness for m in self.members.values()]


@dataclass
class GenomeDistanceCashe:
    distances: dict
    hits: int = 0
    misses: int = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        distance = self.distances[(g0, g1)]
        if distance is None:
            # distance is not already computed
            distance = genome0.distance(genome1)
            self.distances[g0, g1] = distance
            self.distances[g1, g0] = distance
            self.misses += 1
        else:
            self.hits += 1
        return distance

@dataclass
class Species_set:
    config : Config
    indexer : count = count(1)
    species : dict = field(default_factory = dict, metadata={int})
    genome_to_specie :dict =field(default_factory = dict, metadata={str,Specie})