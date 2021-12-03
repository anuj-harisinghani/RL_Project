"""
Collision.py - code provided by WuXinyang2012 user on GitHub
https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057

Using this code to extract collisions of the humanoid with the ground for detecting the steps taken by the robot.
As step length is one of the behavior metrics.
"""

import os
import mujoco_py
import numpy as np
from gym.envs.mujoco.humanoid import HumanoidEnv
import time

# PATH_TO_HUMANOID_XML = os.path.expanduser('~/.mujoco/mujoco200/model/humanoid.xml')
PATH_TO_HUMANOID_XML = os.path.expanduser('/home/anuj/anaconda3/envs/RL/lib/python3.8/site-packages/gym/envs/mujoco/assets/humanoid.xml')

# Load the model and make a simulator
model = mujoco_py.load_model_from_path(PATH_TO_HUMANOID_XML)  # model: class PyMjModel
# sim = mujoco_py.MjSim(model)
# viewer = mujoco_py.MjViewer(sim)
env = HumanoidEnv()


for _ in range(1):
    # sim.reset()
    obs = env.reset()
    sim = env.sim
    # Simulate 1000 steps so humanoid has fallen on the ground
    for _ in range(1000):
        # viewer.render()
        # sim.step()
        env.render()
        env.step(env.action_space.sample())

    print('number of contacts', sim.data.ncon)
    for i in range(sim.data.ncon):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        contact = sim.data.contact[i]
        print('contact:', i)
        print('distance:', contact.dist)
        print('geom1:', contact.geom1, sim.model.geom_id2name(contact.geom1))
        print('geom2:', contact.geom2, sim.model.geom_id2name(contact.geom2))
        print('contact position:', contact.pos)

        # Use internal functions to read out mj_contactForce
        c_array = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)

        # Convert the contact force from contact frame to world frame
        ref = np.reshape(contact.frame, (3, 3))
        c_force = np.dot(np.linalg.inv(ref), c_array[0:3])
        c_torque = np.dot(np.linalg.inv(ref), c_array[3:6])
        print('contact force in world frame:', c_force)
        print('contact torque in world frame:', c_torque)
        print()
