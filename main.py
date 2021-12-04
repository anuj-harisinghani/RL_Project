import os
import numpy as np

from map_elites import MapElites
from map_elites import Individual


# default params for MAPElites
mode = 'default'
n_behaviors = 2
n_niches = 20
bootstrap_archive = None
bootstrap_genome_map = None
map_iterations = 1000
n_init_niches = 50

# default params for Individual
fit_generations = 10
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

bootstrap_archive = burner_map.archive
bootstrap_genome_map = burner_map.genome_map

# check number of initialized cells
indices = np.argwhere(bootstrap_archive != 0)







