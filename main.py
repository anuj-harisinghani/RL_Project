import os
import numpy as np

from map_elites import MapElites
from map_elites import Individual


# default params for MAPElites
n_behaviors = 2
n_niches = 100
bootstrap_archive = None
bootstrap_genome_map = None
map_iterations = 100
n_init_niches = 25

# default params for Individual
fit_generations = 50
dist_threshold = None
n_hidden = 35


# creating bootstrap archive - created with n_behaviours and n_niches
burner_map = MapElites()
burner_map.init_archive()

for i in range(n_init_niches):
    burner_x = Individual(fit_generations, dist_threshold, n_hidden)
    burner_x.init_random_genome()
    burner_x.fit_genome()
    burner_map.update_archive(burner_x)





