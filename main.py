import os
import numpy as np

from map_elites import MapElites
from map_elites import Individual


# default params for MAPElites
mode = 'default'
n_behaviors = 2
# n_niches = 20
bootstrap_archive = None
bootstrap_genome_map = None
map_iterations = 1000
# n_init_niches = 50

# default params for Individual
fit_generations = 10
dist_threshold = None
n_hidden = 75

n_niches = 25
n_cells = n_niches ** n_behaviors
n_init_niches = int(0.4*n_cells)

# creating bootstrap archive - created with n_behaviours and n_niches
burner_map = MapElites(n_niches=n_niches, n_init_niches=n_init_niches)
burner_map.init_archive()

for i in range(n_init_niches):
    burner_x = Individual(fit_generations, dist_threshold, n_hidden)
    burner_x.init_random_genome()
    burner_x.fit_genome()
    burner_map.update_archive(burner_x)

bootstrap_archive = burner_map.archive
bootstrap_genome_map = burner_map.genome_map

# check number of initialized cells
# indices = np.argwhere(bootstrap_archive != 0)


# actual running
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
default_map = MapElites(n_niches=n_niches, n_init_niches=n_init_niches, n_hidden=n_hidden)
default_map.map_algorithm()

from gym.envs.mujoco.humanoid import HumanoidEnv
make_one_action = True
n_actions = 17

# def test_genome(genome, render=True):
#     env = HumanoidEnv()
#     obs = env.reset()
#     rewards = []
#
#     for _ in range(100000):
#         preds = genome.predict(obs.reshape(-1, len(obs)))
#
#         if make_one_action:
#             # preds is a (1, 17) shape vector, choose one action based on softmax
#             action = np.zeros(n_actions)
#             action[preds.argmax()] = 1
#
#         else:
#             action = preds
#
#         # step using the predicted action vector
#         if render:
#             env.render()
#         obs, reward, done, info = env.step(action)
#         rewards.append(reward)
#
#         if done:
#             break
#
#     env.close()
#     return np.mean(rewards)


ar = default_map.archive
gm = default_map.genome_map

network_map = np.empty_like(gm)
for r in range(len(gm)):
    for c in range(len(gm[r])):
        if gm[r][c] != None:
            network_map[r][c] = gm[r][c].genome


indices = np.argwhere(ar != 0)
max_fit_ind = np.argwhere(ar == np.max(ar))

import dill as pickle

with open('./default_arch.pkl', 'wb') as arch:
    pickle.dump(ar, arch)

with open('./default_nm.pkl', 'wb') as nm:
    pickle.dump(network_map, nm)


import matplotlib.pyplot as plt
import matplotlib.colors as colors


fig, ax = plt.subplots()
# bounds = np.array([0, np.min(ar[ar > 0]), np.max(ar)])
norm = colors.TwoSlopeNorm(vmin=0, vcenter=np.min(ar[ar>0]), vmax=np.max(ar))
pcm = ax.pcolormesh(ar, norm=norm)
