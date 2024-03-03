import xml.etree.ElementTree as ET
import os
import numpy as np
import robosuite.utils.transform_utils as tfutil
import copy



def inverseKinematics(DesiredPose_in_U = (np.zeros(3,), np.array([0., 0., 0., 1.])), env = []):
    # These two OPTIONAL helper functions will actually set the angles and get you the gripper endeffector pose and jacobian.    
    #  "getGripperEEFPose" is actually moving the robot in the simulation but it does not render it. This works as a forward kinematics function. If you want to see the new robot pose, add: env.render()
    # "getJacobian(env)" returns the Jacobian computed for the gripper end-effector which is different from what you get in HW3. 

    #getGripperEEFPose(env, setJointAngles)
    #getJacobian(env)


    # We will bring the robot back to original pose at the end of "inverseKinematics" function, because it is inteded to compute the joint angles, not execute the joint angles.
    # But it is not required for you to implement it.




    # Tuple of position and orientation (quat) of the base frame expressed in world frame
    robotBasePose = (env.robots[0].base_pos, env.robots[0].base_ori) 
    initialJointAngles= env.robots[0]._joint_positions
    jointAngles = initialJointAngles.copy()
    
    #============= Your code here =============
    
    epsilon = 0.001
    thetai = jointAngles 
    error = tfutil.get_pose_error(tfutil.pose2mat(DesiredPose_in_U), tfutil.pose2mat(getGripperEEFPose(env, initialJointAngles)))

    if (np.linalg.norm(error) > epsilon):
        error = tfutil.get_pose_error(tfutil.pose2mat(DesiredPose_in_U), tfutil.pose2mat(getGripperEEFPose(env, jointAngles)))
        jointAngles = thetai + np.dot(np.linalg.pinv(getJacobian(env)), error)
        thetai = jointAngles
        #env.render()
        #getGripperEEFPose(env, np.dot(np.linalg.pinv(getJacobian(env)), error6))
    
    #==========================================
    getGripperEEFPose(env, initialJointAngles) # Brings the robot to the initial joint angle.
    env.render()
    return np.append(jointAngles, 0)










#=========== Not a HW problem below ==========

def getGripperEEFPose(env, setJointAngles): # This function works as a forward Kinematics

    #env.robots[0].set_robot_joint_positions(setJointAngles)
    gripper_EEF_pose = (env.robots[0].sim.data.get_body_xpos('gripper0_eef'), tfutil.convert_quat(env.robots[0].sim.data.get_body_xquat('gripper0_eef')))     
    return gripper_EEF_pose # Outputs the position and quaternion (x,y,z,w) of the EEF pose in Universial Frame{0}.

def getJacobian(env): # This function returns the jacobian of current configurations
    jacp = env.robots[0].sim.data.get_body_jacp('gripper0_eef').reshape((3, -1))[:,env.robots[0]._ref_joint_vel_indexes]
    jacr = env.robots[0].sim.data.get_body_jacr('gripper0_eef').reshape((3, -1))[:,env.robots[0]._ref_joint_vel_indexes]    
    jacobianMat_gripperEEF = np.concatenate((jacp, jacr),axis=0)
    return jacobianMat_gripperEEF #Outputs the Jacobian expressed in {0}

def discretize_state(state, precision):
    # Assuming state_space is the concatenated state space
    # and num_bins_per_dim is the number of bins per dimension
    discretized_state = []
    for i in range(len(state)):
        discrete_value = int(state[i] % precision)
        discretized_state.append(discrete_value)
    return tuple(discretized_state)  # Convert to tuple for use as dictionary key

def discretize_dimension(value, num_bins):
    bin_width = 1.0 / num_bins
    return min(int(value / bin_width), num_bins - 1)

def discretize_state(state, num_bins_per_dimension):
    state_index = 0
    current_index_offset = 0
    for i, value in enumerate(state):
        num_bins = num_bins_per_dimension[i]
        dimension_index = discretize_dimension(value, num_bins)
        state_index += dimension_index * (num_bins_per_dimension[i] ** i)
    return state_index

def discretize_action_dimension(value, num_bins):
    """
    Discretize a single dimension of the action space into a given number of bins.
    
    Args:
        value: The value to discretize.
        num_bins: The number of bins to discretize the dimension into.
    
    Returns:
        The index of the bin that the value falls into.
    """
    bin_width = 1.0 / num_bins
    return min(int(value / bin_width), num_bins - 1)

def discretize_action(action, num_bins_per_dimension):
    """
    Discretize the entire action space into a given number of bins per dimension.
    
    Args:
        action: The action array.
        num_bins_per_dimension: The number of bins to discretize each dimension into.
    
    Returns:
        The discretized action array.
    """
    discretized_action = []
    for i, value in enumerate(action):
        num_bins = num_bins_per_dimension[i]
        discretized_value = discretize_action_dimension(value, num_bins)
        discretized_action.append(discretized_value)
    return discretized_action