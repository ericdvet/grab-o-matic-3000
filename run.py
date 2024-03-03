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

# Define RL parameters
num_episodes = 1000
max_steps_per_episode = 100

# Define Q-learning parameters
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Initialize Q-values
state_space_size = 10 ** 3
action_space_size = 3 ** 8
q_table = np.zeros((state_space_size, action_space_size)).astype(np.float32)

# reset the environment
env.reset()
obs = None

ball_fall_delay = 0
ball_fall_flag = False

# env.robots[0].set_robot_joint_positions([1, 1, 1, 1, 1, 1, 1])

low, high = env.action_spec
action = np.random.uniform(low, high)
obs, reward, done, info = env.step(action)  # take action in the environment

state_index = discretize_state(obs.get("robot0_eef_pos"))

for episode in range(num_episodes):
    for oui in range(1000):

        """action = model(env, obs) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment

        # odict_keys(['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'ball_pos', 'ball_quat', 'gripper_to_ball_pos', 'robot0_proprio-state', 'object-state'])
           
        print(obs.get("robot0_eef_pos"))"""

        if np.random.uniform(0, 1) > exploration_rate:
            action_index = np.argmax(q_table[state_index, :])
            action = action_index_to_action(action_index)
        else:
            low, high = env.action_spec
            action = np.random.uniform(low, high)
            action_index = discretize_action(action)
            
        obs, reward, done, info = env.step(action)

        pos = obs.get("robot0_eef_pos")
        next_state_index = discretize_state(pos)

        # Update Q-value using Q-learning equation
        old_q_value = q_table[state_index, action_index]
        next_max_q_value = np.max(q_table[next_state_index, :])
        new_q_value = old_q_value + learning_rate * (reward + discount_rate * next_max_q_value - old_q_value)
        q_table[state_index, action_index] = new_q_value

        state_index = next_state_index

        #if (env._check_failure()):
            #env.resetBallPosition()

        """if ball_fall_flag == False:
            ball_fall_delay += 1

        if ball_fall_delay == 100:
            ball_fall_flag = True
            ball_fall_delay = 0
            env.resetBallPosition()"""
            
        env.render()  # render on display 
     # Decay exploration rate
    env.reset()
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
env.render()  # render on display 

#print("Timesteps taken: {}".format(epochs))
#print("Penalties incurred: {}".format(penalties))

while (1):
    1