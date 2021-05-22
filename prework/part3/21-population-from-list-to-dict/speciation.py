

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


def genetic_chromosome_distance(genome1_chromsome, genome2_chromsome):
    # Compute node gene distance component.
    disjoint_genes = 0
    ant_homologous_genes = 0
    total_distance_from_matching_genes = 0
    chromosome1_keys = genome1_chromsome.keys()
    chromosome2_keys = genome1_chromsome.keys()
    for k2 in chromosome2_keys:
        if k2 not in chromosome1_keys:
            disjoint_genes += 1

    for k1 in chromosome1_keys:
        if k1 not in chromosome2_keys:
            disjoint_genes += 1
        else:
            # Homologous genes compute their own distance value.
            ant_homologous_genes += 1
            total_distance_from_matching_genes += genome1_chromsome[k1].gene_distance(genome2_chromsome[k1])

    max_genes = max(len(chromosome1_keys), len(chromosome2_keys))
    # How similar compared to how similar it possible could have been by disjoint genes
    distance_from_disjoin_genes = (disjoint_genes * gene_compatibility_disjoint_coefficient) / max_genes
    average_homologous_gene_distance = total_distance_from_matching_genes/ ant_homologous_genes
    total_chromosome_distance = distance_from_disjoin_genes + average_homologous_gene_distance
    return total_chromosome_distance

@dataclass
class GenomeDistanceCashe:
    distances : dict = field(default_factory=dict)
    hits : int = 0
    misses : int = 0

    def __call__(self, genome0,genome1):
        g0 = genome0.key
        g1 = genome1.key

        if (g0, g1) in self.distances:
            self.hits +=1
        else:
            self.misses +=1
            d = get_genome_distance(genome0,genome1)
            self.distances[(g0, g1)] = d
            self.distances[(g1, g0)] = d
        return d


@dataclass
class Spicie:
    key : int
    created : int
    last_improved : int #= field(created)#= created
    representative = None
    members : dict = field(default_factory=dict)
    fitness : float = None
    adjusted_fitness = None
    fitness_history : list = field(default_factory=list)

    def __post_init__(self):
        self.last_improved = self.created

    def update(self, representative, members):
        self.representative = representative
        self.members = members


    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]


def speciate_population(population: dict):
    compatibility_threshold


    # Find the best representatives for each existing species.
    unspeciated = {individual.id for individual in population}
    new_representatives = {}
    new_members = {}
    # First find new representatives in case old rep died. This can move specie center closer or furhter apart
    # If moves closer, can snack up members, if move futher apart , could loose members so that they become their own species
    for s_id, specie in species.items():
        candidates  =[]
        for gid in unspeciated:
            g = population[gid]
            d = genomeDistanceCashe[specie.representative,g]
            candidates.append((d,g))

        # The new representative is the genome closest to the current representative.
        ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
        new_rep_id = new_rep.key
        new_representatives[s_id] =new_rep_id # hvorfor ny representant? i tilfelle forrige døde
        unspeciated.remove(new_rep_id)

    # Now that we know that all species rep actually are alive, find members within treshold distance
    while unspeciated:
        individ_id = unspeciated.pop()
        individ = population[individ]
        # population needs to become a dict! # othervise search through it for every new rep



def get_genome_distance(genome1, genome2):
    chromosome_distances = {}
    for chromosome_key, genome1_chromsome in vars(genome1).items():
        genome2_chromsome = genome2.get(chromosome_key)
        chromosome_distances[chromosome_key] = genetic_chromosome_distance(genome1_chromsome,genome2_chromsome)

    distance = sum(chromosome_distances.values())/len(chromosome_distances)
    print("chromosome distances {}: {}".format(distance, chromosome_distances))
    return distance

