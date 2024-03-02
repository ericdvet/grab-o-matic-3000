import numpy as np

import robosuite
from env import NewLift
from model import NaiveModel

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

while not env._check_success():
    action = model(env, obs) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment

    print(reward)

    if env._check_failure():
        ball_fall_flag = False
        penalties += 1

    if ball_fall_flag == False:
        ball_fall_delay += 1

    if ball_fall_delay == 100:
        ball_fall_flag = True
        ball_fall_delay = 0
        env.resetBallPosition()
        
    epochs += 1
    env.render()  # render on display 
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))
env.render()  # render on display 

#print("Timesteps taken: {}".format(epochs))
#print("Penalties incurred: {}".format(penalties))

while (1):
    1