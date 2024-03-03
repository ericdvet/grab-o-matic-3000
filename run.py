import numpy as np

import robosuite
from env import NewLift
from model import NaiveModel
import random

from kinematics import *

# initialize environment
robots = "Panda"
env = robosuite.make(
    'NewLift',
    robots,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    render_camera="sideview"
)

# initailiza model (random model here)
model = NaiveModel(env.robots[0].dof)

# reset the environment
env.reset()
obs = None

ball_fall_delay = 0
ball_fall_flag = False

# q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

total_epochs, total_penalties = 0, 0
episodes = 100

epochs = 0
penalties, reward = 0, 0
done = False

q_table = np.zeros([5 ** 6, 5    ** 6])

low, high = env.action_spec
action = np.random.uniform(low, high)
obs, reward, done, info = env.step(action)  # take action in the environment

state_space = np.concatenate((obs.get("robot0_eef_pos"),
                                    obs.get("ball_pos")))
num_bins_per_dimension = [5, 5, 5, 5, 5, 5]
state_index = discretize_state(state_space, num_bins_per_dimension)


num_bins_per_dimension_action = [5, 5, 5, 5, 5, 5]

for i in range(1, 100):
    while not env._check_success():
        
        action = model(env, obs) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
                                    
        """if random.uniform(0, 1) < epsilon:
            low, high = env.action_spec
            action = np.random.uniform(low, high)
        else:
            state_index = discretize_state(state_space, num_bins_per_dimension)
            action = np.argmax(q_table[state_index]) # Exploit learned values

        obs, reward, done, info = env.step(action)  # take action in the environment

        discretized_action = discretize_action(action, num_bins_per_dimension_action)

        state_space = np.concatenate((obs.get("robot0_eef_pos"), 
                                    obs.get("ball_pos")))
        state_index = discretize_state(state_space, num_bins_per_dimension)


        old_value = q_table[state_index, discretized_action]
        next_max = np.max(q_table[state_index])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_index, discretized_action] = new_value
        
        if reward < -1:
            penalties += 1"""

        if ball_fall_flag == False:
            ball_fall_delay += 1

        if ball_fall_delay == 100:
            ball_fall_flag = True
            ball_fall_delay = 0
            env.resetBallPosition()
        
        """if env._check_failure():
            ball_fall_flag = False
            penalties += 1"""
            
        epochs += 1
        env.render()  # render on display 
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))
env.render()  # render on display 

#print("Timesteps taken: {}".format(epochs))
#print("Penalties incurred: {}".format(penalties))

while (1):
    1