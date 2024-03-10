"""ball_catching controller."""

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from controller import Robot
from controller import Supervisor
from controller import Keyboard

from gym.spaces import Box, Discrete
import numpy as np
import time as t
from time import time
from math import *
from scipy.spatial.transform import Rotation as R
import random

import robotic_manipulations as tae
import simulation_gen as sim

class GrabberRobot(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        
        # observation space
        # eef pos x [-1 1]
        # eef pos y [-1 1]
        # eef pos z [-1 1]
        self.observation_space = Box(low=np.array([-1, -1, -1]),
                                     high=np.array([-1, -1, -1]),
                                     dtype=np.float64)
        
        # action space
        # joint angles 1 - 7
        self.action_space = Box(low=np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
                                high=np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
                                dtype=np.float64)
        
        supervisor = Supervisor()
        # get the time step of the current world.
        timestep = int(supervisor.getBasicTimeStep())

        motorNames = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', \
                'wrist_2_joint', 'wrist_3_joint']
        motorDevices = []
        motor_ang = []
        for motorName in motorNames: 
            motor = supervisor.getDevice(motorName)
            motor.setPosition(0.0)
            motor.setVelocity(motor.getMaxVelocity()) #move at max speed
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(timestep)
            motorDevices.append(motor)
            motor_ang.append(motor.getPositionSensor().getValue())
        
        jacobian = tae.create_jacobian(motor_ang)
        H = tae.generate_final_transform(motor_ang)
        x_ee,y_ee,z_ee = H[0,3], H[1, 3], H[2, 3]
        yaw, pitch, roll = tae.euler_from_Htrans(H)
        current = [x_ee,y_ee,z_ee,yaw,pitch,roll]

        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        # Position on x-axis
        eef_x_pos = normalize_to_range(x_ee, -np.pi, np.pi, -1.0, 1.0)
        eef_y_pos = normalize_to_range(y_ee, -np.pi, np.pi, -1.0, 1.0)
        eef_z_pos = normalize_to_range(z_ee, -np.pi, np.pi, -1.0, 1.0)
        return [eef_x_pos, eef_y_pos, eef_z_pos]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]
    
    def get_reward(self, action=None):
        return 1

    def is_done(self):
        return sim.isTouched()

    # needs redo
    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False
    
    def get_info(self):
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        # Set motor velocities with consideration for motor limits
        for i, motor in enumerate(self.wheels):
            if abs(action[i]) > motor.getMaxVelocity():
                vel = (motor.getMaxVelocity() - 0.0001) * action[i] / abs(action[i])
                motor.setVelocity(vel)
            else:
                motor.setVelocity(action[i])

# it starts here :O
                
env = GrabberRobot()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)

solved = False

episode_count = 0
episode_limit = 2000

# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")

        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()