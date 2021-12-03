"""
Collision.py - code provided by WuXinyang2012 user on GitHub
https://gist.github.com/WuXinyang2012/b6649817101dfcb061eff901e9942057

Using this code to extract collisions of the humanoid with the ground for detecting the steps taken by the robot.
As step length is one of the behavior metrics.
"""

import os
import math
import numpy as np
from gym.envs.mujoco.humanoid import HumanoidEnv
import time

env = HumanoidEnv()
sim = env.sim
# viewer = env._get_viewer('human')

for _ in range(1):
    # something = []
    obs2 = env.reset()
    # env.render()

    # init step behaviour values
    step_dist_origin = []
    step_dist_between = []
    foot_timestep = 0
    foot_pos = np.array([0.0] * 3)
    dist = 0.0

    for iteration in range(100):

        # viewer.render()
        # time.sleep(1)
        # env.render()
        action = env.action_space.sample()
        # action = np.random.randint(-5, 5)
        # position = np.random.randint(0, len(sim.data.ctrl))
        # sim.data.ctrl[position] = action
        # sim2.data.ctrl[position] = action
        # sim.data.ctrl[:] = action
        # sim.step()
        # sim2.step()

        # data = sim.data
        # obs = np.concatenate(
        #     [
        #         data.qpos.flat[2:],
        #         data.qvel.flat,
        #         data.cinert.flat,
        #         data.cvel.flat,
        #         data.qfrc_actuator.flat,
        #         data.cfrc_ext.flat,
        #     ])
        # obs2 = env._get_obs()  # same as obs from sim.data
        obs, r, done, info = env.step(action)
        # env.render()

        for i in range(len(sim.data.contact)):
            contact = sim.data.contact[i]

            '''
            contact ids:
            floor = 0
            left foot = 11
            right foot = 8
            '''

            # initial filter - keeping contacts with the floor and removing all others
            geom_list = [contact.geom1, contact.geom2]
            if contact.geom1 != contact.geom2 and 0 in geom_list:
                # check for any foot touching
                # if contact.geom1 == 0 and (contact.geom2 == 11 or contact.geom2 == 8):
                if 11 in geom_list or 8 in geom_list:
                    print(iteration, 'foot touched on the ground!')
                    # print(geom_list)

                    delta_distance_from_origin = contact.dist - dist
                    delta_distance_between_steps = math.dist(contact.pos, foot_pos)
                    # if iteration > foot_timestep and delta_distance_from_origin > 0:
                    if iteration > foot_timestep:
                        print(geom_list)
                        print('pos', contact.pos, foot_pos, delta_distance_between_steps)
                        print('dist', contact.dist, dist, delta_distance_from_origin)
                        # append changes in step distances
                        step_dist_origin.append(delta_distance_from_origin)
                        step_dist_between.append(delta_distance_between_steps)

                        # update previous distance and position values
                        dist = contact.dist
                        foot_pos = contact.pos
                        foot_timestep = iteration

                        print('updated timesteps and position: ', foot_timestep, dist)

    step_dist_origin = np.array(step_dist_origin)
    step_dist_between = np.array(step_dist_between)

                        # # one by one foot touching
                        # if contact.geom1 == 0 and contact.geom2 == 11:
                        #     left_foot_pos = contact.pos
                        #     continue
                        # elif contact.geom1 == 0 and contact.geom2 == 8:
                        #     right_foot_pos = contact.pos
                        #     continue


# if os.name == 'posix':
#     PATH_TO_HUMANOID_XML = os.path.expanduser(
#         '/home/anuj/anaconda3/envs/RL/lib/python3.8/site-packages/gym/envs/mujoco/assets/humanoid.xml')
#
# else:
#     PATH_TO_HUMANOID_XML = os.path.expanduser(
#         r'C:\Users\Anuj\Anaconda3\envs\RL\Lib\site-packages\gym\envs\mujoco\assets\humanoid.xml')
