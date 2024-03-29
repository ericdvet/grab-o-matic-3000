"""move_joint controller."""
import time as t
from time import time
from controller import Robot
from controller import Supervisor
from controller import Node
from controller import Keyboard
from controller import Lidar
import numpy as np
from math import *
from scipy.spatial.transform import Rotation as R
import random
import torch
import torch.nn as nn
import joblib
from model import ImitationModel, ImitationModelFiveLayers

DEMO = False
demo_iterator = 0
success_ball_info = [[-1.3704633246986604, -3.7579023770789863, 1, 1.037752345407003, 2.2266142389942374, 9.83105031343256],
    [3.988314011787137, 0.3055345207703928, 1, -2.3345027560678413, -0.4983019634587256, 9.892886973323831],
                     [1.3865763613851851, 3.751986939482575, 1, -1.0601001341628018, -1.534201462351282, 9.811307809100205],
                     [-1.4762112629027628, -3.7176336972971167, 1, 0.38484393822263496, 2.2101807447639925, 9.919877291223392]]

succes_bal_targets = [[0.6890058150478973, -0.7122490557570307, 1.0932401304097925],
                        [-0.680691500348546, -0.6910694061470585, 1.1657739466476615],
                      [-0.7336239069404186, 0.6835840147800106, 1.0026156182004085],
                      [-0.7065233864574929, 0.7027277922308683, 1.2197545824467837]]

# Variable determing amount of camera frames to capture
NUM_FRAMES = 10

# Important controller variables
LEARNING = True
gravity = 9.81

# Generate the position of the ball in the Webots simulation environment.
# Paramters: None
# Returns:
#       list: A list containing the x, y, and z coordinates of the ball position.
def genBallPos():
    ballNode = supervisor.getFromDef('ball')
    translation_field = ballNode.getField('translation')

    angle = random.uniform(0, 2 * pi)
    height = 1
    radius = 4
    x = radius * cos(angle)
    y = radius * sin(angle)
    if DEMO:
        x = success_ball_info[demo_iterator][0]
        y = succes_bal_targets[demo_iterator][1]
    startingPosition = [x, y, height]
    translation_field.setSFVec3f(startingPosition)
    return startingPosition

# Function to launch a ball from a random position towards a random target point within the robot's reachable area
# Parameters: None
# Returns: Tuple containing:
#       List of velocities [velx, vely, velz] to launch the ball towards the target
#       List of target coordinates [targetX, targetY, targetZ]
def launchBall():
    tof = 2 # random.uniform(1,2)
    ballNode = supervisor.getFromDef('ball')
    maxRobotReach = 0.8
    minRobotReach = 0.6
    if random.choice([True, False]):
        targetX = random.uniform(-maxRobotReach, -minRobotReach)
    else:
        targetX = random.uniform(minRobotReach, maxRobotReach)

    if random.choice([True, False]):
        targetY = random.uniform(-maxRobotReach, -minRobotReach)
    else:
        targetY = random.uniform(minRobotReach, maxRobotReach)

    targetZ = random.uniform(1, 1.25)

    if DEMO:
        targetX = succes_bal_targets[demo_iterator][0]
        targetY = succes_bal_targets[demo_iterator][1]
        targetZ = succes_bal_targets[demo_iterator][2]


    translation_field = ballNode.getField('translation')
    x, y, z = translation_field.getSFVec3f()
    velx = (targetX - x) / tof
    vely = (targetY - y) / tof
    velz = (targetZ - z + (1 / 2) * gravity * tof**2) / tof
    ballNode.setVelocity([velx, vely, velz, 0, 0, 0])
    ballNode = None
    return ([velx, vely, velz], [targetX, targetY, targetZ])

# Function to determine if the ball is touched by the robot
# Parameters:
#       robotPos: List containing the position of the robot [x, y, z]
# Returns:
#       Boolean value indicating whether the ball is touched by the robot (True) or not (False)
def isTouched(robotPos):
    ballNode = supervisor.getFromDef('ball')
    trans_field = ballNode.getField("translation")
    diff = np.zeros(3)
    for i in range(3):
        diff[i] = abs(trans_field.getSFVec3f()[i] - robotPos[i])
    if np.linalg.norm(diff) < 0.1:
        return True
    else:
        return False

# Function to convert rotation represented as axis-angle to Euler angles
# Parameters:
#       rot (list): List containing the rotation parameters [x, y, z, angle]
# Returns:
#       list: List of Euler angles [roll, yaw, pitch]
def axis_euler(rot):
    x, y, z, angle = rot
    mag = sqrt(x*x + y*y + z*z)
    x /= mag
    z /= mag
    y /= mag
    yaw = atan2(y * sin(angle) - x*z*(1-cos(angle)), 1 - (y**2 + z**2 ) * (1 - cos(angle)))
    pitch = asin(x * y * (1 - cos(angle)) + z * sin(angle))
    roll = atan2(x * sin(angle)-y * z * (1 - cos(angle)) , 1 - (x**2 + z**2) * (1 - cos(angle)))
    return [-roll, -yaw, pitch]

# Function to convert Euler angles to axis-angle representation
# Parameters:
#       euler (list): List of Euler angles [roll, yaw, pitch]
# Returns:
#       list: List containing the axis-angle representation [x, y, z, angle]
def euler_axis(euler):
    c1 = cos(euler[0] / 2)
    c2 = cos(euler[1] / 2)
    c3 = cos(euler[2] / 2)
    s1 = sin(euler[0] / 2)
    s2 = sin(euler[1] / 2)
    s3 = sin(euler[2] / 2)
    
    x = s1 * s2 * c3 + c1 * c2 * s3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    
    angle = 2*acos(c1 * c2 * c3 - s1 * s2 * s3)
    
    return[x, y, z, angle]

# Function to generate the homogeneous transformation matrix for a given set of DH parameters
# Parameters:
#       theta (float): The joint angle in radians
#       a (float): The link length in meters
#       d (float): The link offset in meters
#       alpha (float): The twist angle in radians
# Returns:
#       numpy.matrix: Homogeneous transformation matrix
def generate_H(theta, a, d, alpha):
    return np.asmatrix([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)], 
                        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)], 
                        [0, sin(alpha), cos(alpha), d], 
                        [0, 0, 0, 1]])

# Function to generate the final transformation matrix for a robotic arm
# Parameters:
#       theta_vals (list): List of joint angles in radians
# Returns:
#       numpy.matrix: Final transformation matrix
def generate_final_transform(theta_vals):
    alpha_vals = [pi/2, 0, 0, pi/2, -pi/2, 0]
    a_vals = [0, -0.6127, -0.57155, 0, 0, 0]
    d_vals = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]

    H_1 = generate_H(theta_vals[0], a_vals[0], d_vals[0], alpha_vals[0])
    H_2 = generate_H(theta_vals[1], a_vals[1], d_vals[1], alpha_vals[1])
    H_3 = generate_H(theta_vals[2], a_vals[2], d_vals[2], alpha_vals[2])
    H_4 = generate_H(theta_vals[3], a_vals[3], d_vals[3], alpha_vals[3])
    H_5 = generate_H(theta_vals[4], a_vals[4], d_vals[4], alpha_vals[4])
    H_6 = generate_H(theta_vals[5], a_vals[5], d_vals[5], alpha_vals[5])

    H_1_2 = H_1 @ H_2
    H_1_3 = H_1_2 @ H_3
    H_1_4 = H_1_3 @ H_4
    H_1_5 = H_1_4 @ H_5
    H_1_6 = H_1_5 @ H_6

    return H_1_6    

# Function to create the Jacobian matrix for a robotic arm
# Parameters:
#       theta_vals (list): List of joint angles in radians
# Returns:
#       numpy.array: The Jacobian matrix
def create_jacobian(theta_vals):
    alpha_vals = [pi/2, 0, 0, pi/2, -pi/2, 0]
    a_vals = [0, -0.6127, -0.57155, 0, 0, 0]
    d_vals = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]

    H_1 = generate_H(theta_vals[0], a_vals[0], d_vals[0], alpha_vals[0])
    H_2 = generate_H(theta_vals[1], a_vals[1], d_vals[1], alpha_vals[1])
    H_3 = generate_H(theta_vals[2], a_vals[2], d_vals[2], alpha_vals[2])
    H_4 = generate_H(theta_vals[3], a_vals[3], d_vals[3], alpha_vals[3])
    H_5 = generate_H(theta_vals[4], a_vals[4], d_vals[4], alpha_vals[4])
    H_6 = generate_H(theta_vals[5], a_vals[5], d_vals[5], alpha_vals[5])

    H_1_2 = H_1 @ H_2
    H_1_3 = H_1_2 @ H_3
    H_1_4 = H_1_3 @ H_4
    H_1_5 = H_1_4 @ H_5
    H_1_6 = H_1_5 @ H_6

    r_0_0 = np.matrix([0,0,1]).astype('float')
    r_0_1 = np.transpose(H_1[0:3, 2])
    r_0_2 = np.transpose(H_1_2[0:3, 2])
    r_0_3 = np.transpose(H_1_3[0:3, 2])
    r_0_4 = np.transpose(H_1_4[0:3, 2])
    r_0_5 = np.transpose(H_1_5[0:3, 2])

    d_0_0 = np.transpose(np.matrix([[0], [0], [0]]))
    d_0_1 = np.transpose(H_1[0:3, 3])
    d_0_2 = np.transpose(H_1_2[0:3, 3])
    d_0_3 = np.transpose(H_1_3[0:3, 3])
    d_0_4 = np.transpose(H_1_4[0:3, 3])
    d_0_5 = np.transpose(H_1_5[0:3, 3])
    d_0_6 = np.transpose(H_1_6[0:3, 3])


    linear_jacobian = np.concatenate( (np.transpose(np.cross(r_0_0, d_0_6)), np.transpose(np.cross(r_0_1, (d_0_6-d_0_1))), \
        np.transpose(np.cross(r_0_2, (d_0_6-d_0_2))),np.transpose(np.cross(r_0_3, (d_0_6-d_0_3))), \
            np.transpose(np.cross(r_0_4, (d_0_6-d_0_4))), np.transpose(np.cross(r_0_5, (d_0_6-d_0_5)))), axis = 1)

    angular_jacobian = np.concatenate((np.transpose(r_0_0), np.transpose(r_0_1), np.transpose(r_0_2), np.transpose(r_0_3), \
        np.transpose(r_0_4), np.transpose(r_0_5)), axis = 1)
                                                               
    jacobian = np.concatenate((linear_jacobian, angular_jacobian), axis = 0)
    return jacobian

# Function to calculate the error between the given and goal positions and orientations
# Parameters:
#       given (list): List containing the given position and orientation [x, y, z, yaw, pitch, roll]
#       goal (list): List containing the goal position and orientation [x_pos, y_pos, z_pos, x_axis, y_axis, z_axis, angle]
# Returns:
#       numpy.array: Error vector
def calculate_error(given, goal):
    # Given is expected to be [x, y, z, yaw, pitch, roll]
    # Goal is given in [x_pos, y_pos, z_pos, x_axis, y_axis, z_axis, angle]
    euler_angles = axis_euler(goal[3:])
    return np.array([goal[0] - given[0],goal[1] - given[1],goal[2] - given[2] - 0.6, euler_angles[0] - given[3], euler_angles[1] - given[4], euler_angles[2] - given[5]])

def euler_from_Htrans(H):
    # beta = -np.arcsin(H[2,0])
    # alpha = np.arctan2(H[2,1]/np.cos(beta),H[2,2]/np.cos(beta))
    # gamma = np.arctan2(H[1,0]/np.cos(beta),H[0,0]/np.cos(beta))
    # return [alpha, beta, gamma]
    r = R.from_matrix(H[0:3, 0:3])
    r = r.as_euler('xyz', degrees = False)
    r = [r.item(0), r.item(1), r.item(2)]
    return r

# Function to calculate the joint velocities based on the error and Jacobian matrix
# Parameters:
#       error (numpy.array): Error vector
#       jacobian (numpy.array): The Jacobian matrix
# Returns:
#       numpy.array: Joint velocities
def calculate_joint_vel(error, jacobian):
    kPt = 200
    kPa = 7.5
    # kPt = 3
    # kPa = 1
    scaled_error = np.concatenate((error[:3] * kPt, error[3:] * kPa), axis=0)
    # return np.linalg.inv(jacobian) @ scaled_error
    return scaled_error @ jacobian

# ============================== Main ============================== 

supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())

# Get the target position
target = supervisor.getFromDef('TARGET')
translation_field = target.getField('translation')
rotation_field = target.getField('rotation')
target = super

# Initialize robot joints
print('Using timestep: %d' % timestep)
motorNames = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', \
                'wrist_2_joint', 'wrist_3_joint']
motorDevices = []
for motorName in motorNames: 
    motor = supervisor.getDevice(motorName)
    motor.setPosition(0.0)
    motor.setVelocity(motor.getMaxVelocity()) #move at max speed
    position_sensor = motor.getPositionSensor()
    position_sensor.enable(timestep)
    motorDevices.append(motor)

# Initial position
initialPos = [0, -1.57, 0.0, 0.0, 0.0, 0.0]
for i in range(len(motorDevices)):
    motorDevices[i].setPosition(initialPos[i])

#Set up the first two joints for keyboard vel control
motionAxis = 0
#Enable vel control for the pan:
for i in range(len(motorDevices)):
    motorDevices[i].setPosition(float('+inf'))
    motorDevices[i].setVelocity(0.0)

# Initialize data collection variables
observations = []
actions = []
outcomes = []

# Initialize misc. simulation variables
currentTime = 0
ballLaunched = False
base_time = supervisor.getTime()
numRuns = 0
ball_caught = False

if not LEARNING:
    
    # Load the trained model
    model = ImitationModelFiveLayers(input_size=NUM_FRAMES * 4, output_size=7)
    model.load_state_dict(torch.load('imitation_model_five_layers.pth'))
    model.eval()
    
    # Load scalar (not sure if necessary?)
    scaler = joblib.load('scaler.pkl')

sucker = supervisor.getDevice("vacuum gripper")
sucker.turnOn()
sucker.enablePresence(timestep)

c = supervisor.getDevice("camera")
c.enable(timestep)
c.recognitionEnable(timestep)

c2 = supervisor.getDevice("camera2")
c2.enable(timestep)
c2.recognitionEnable(timestep)

time_intervals = [0.05, 0.1 ,0.15, 0.2, 0.25]
cam_info = []
for i in range(NUM_FRAMES):
    cam_info.extend([0, 0, 0, 0])

frame_count = 0
has_frames = False

while supervisor.step(timestep) != -1:
    
    # Shoot ball towards robot arm at regular intervals
    if not ballLaunched:
        ballX, ballY, ballZ = genBallPos()
        vels, targetPos = launchBall()
        demo_iterator += 1
        if demo_iterator == len(succes_bal_targets):
            demo_iterator = 0
        ball_initial_info = [ballX, ballY, ballZ, vels[0], vels[1], vels[2]]
        print('\nTrial ', numRuns, end = '')
        ballLaunched = True
        lastLaunch = currentTime
        has_frames = False
        if LEARNING:
            c.enable(timestep)
            c2.enable(timestep)

    if (currentTime - lastLaunch) < 3.5:
        translation_field.setSFVec3f(targetPos)
        if (currentTime - lastLaunch) > 0.1:
            if not has_frames:
                balls = c.getRecognitionObjects()
                balls2 = c2.getRecognitionObjects()
                cam_info[4*frame_count:4*frame_count+2] = balls[0].getPositionOnImage()[:2]
                cam_info[4*frame_count+2:4*frame_count+4] = balls2[0].getPositionOnImage()[:2]

                frame_count+=1
                if frame_count == NUM_FRAMES:
                    if False:
                        c.disable()
                        c2.disable()
                    frame_count = 0
                    has_frames = True
    elif (currentTime - lastLaunch) < 3.6:
        ballNode = supervisor.getFromDef('ball')
        ball_translation_field = ballNode.getField('translation')
        startingPosition = [0, 0, -1]
        ball_translation_field.setSFVec3f(startingPosition)
    else:
        ballLaunched = False
        numRuns += 1
        if ball_caught:
            print('ball info: ' + str(ball_initial_info))
            print('target pos: ' + str(targetPos))
            print(" Success", end = '')
            if LEARNING:
                # print(cam_info)
                # print(len(cam_info))
                obs = np.copy(cam_info)
                acts = np.copy(goal)
                outcomes.append(ball_caught)
                actions.append(acts)
                observations.append(obs)
        ball_caught = False
            
    currentTime = supervisor.getTime()
    
    # Calculate current motor location
    motor_ang = []
    for motor in motorDevices:
        motor_ang.append(motor.getPositionSensor().getValue())
        
    jacobian = create_jacobian(motor_ang)
    H = generate_final_transform(motor_ang)
    x_ee,y_ee,z_ee = H[0,3], H[1, 3], H[2, 3]
    yaw, pitch, roll = euler_from_Htrans(H)
    current = [x_ee,y_ee,z_ee,yaw,pitch,roll]
        
    if LEARNING:
        # Calculate error based upon calculated error
        goal = translation_field.getSFVec3f()
        goal[0] *= -1
        goal[1] *= -1
        goal.extend(rotation_field.getSFRotation())
        error = calculate_error(current, goal)
        # Use inverse kinematics to catch ball
        joint_vel = calculate_joint_vel(error, jacobian)
        i = 0
        for motor in motorDevices:
            # print(joint_vel.item(i))
            if abs(joint_vel.item(i)) > motor.getMaxVelocity():
                vel = (motor.getMaxVelocity() - 0.0001) * joint_vel.item(i) / abs(joint_vel.item(i))
                motor.setVelocity(vel)
            else:
                motor.setVelocity(joint_vel.item(i))
            i += 1
    else:
        # Preprocess observations for model
        if(has_frames):
            new_observations = np.array(cam_info)
            single_observation_scaled = scaler.transform(new_observations.reshape(1,-1))
            new_observation_tensor = torch.tensor(single_observation_scaled, dtype=torch.float32)
            
            # Use the trained model to predict actions based on the new observations
            with torch.no_grad():
                predicted_goal = model(new_observation_tensor)
            predicted_goal = predicted_goal.tolist()
            error = calculate_error(current, predicted_goal[0])

            # Use inverse kinematics to catch ball
            joint_vel = calculate_joint_vel(error, jacobian)
            i = 0
            if not ball_caught:
                for motor in motorDevices:
                    # print(joint_vel.item(i))
                    if abs(joint_vel.item(i)) > motor.getMaxVelocity():
                        vel = (motor.getMaxVelocity() - 0.0001) * joint_vel.item(i) / abs(joint_vel.item(i))
                        motor.setVelocity(vel)
                    else:
                        motor.setVelocity(joint_vel.item(i))
                    i += 1
            else:
                for i, motor in enumerate(motorDevices):
                    motor.setVelocity(0)
    
    # Check if ball is caught
    robot_pos = [-x_ee, -y_ee, z_ee + 0.6]
    if isTouched(robot_pos):
        sucker.turnOn()
        if sucker.getPresence():
            ball_caught = True
            ballNode = supervisor.getFromDef('ball')
            ball_translation_field = ballNode.getField('translation')
            startingPosition = [0, 0, -1]
            # ball_translation_field.setSFVec3f(startingPosition)
            # ballNode.setVelocity([0, 0, -1, 0, 0, 0])
    else:
        sucker.turnOff()
    
    # Store observation and action data
    if LEARNING:
        if (numRuns == 10000):
            np.savez("observations.npz", observations)
            np.savez("actions.npz", actions)
            np.savez("outcomes.npz", outcomes)
            break

# Enter here exit cleanup code.
for i in range(len(motorDevices)):
    motorDevices[i].setVelocity(0.0)