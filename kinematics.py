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

def discretize_state(position, num_position_bins=10, min_position=[-1.0, -1.0, 0], max_position=[1.0, 1.0, 2]):
    state_index = 0

    for i in range(len(position)):
        bin_index = int(np.clip((position[i] - min_position[i]) / (max_position[i] - min_position[i]) * num_position_bins, 0, num_position_bins - 1))
        state_index += bin_index * (num_position_bins ** (len(position) - 1 - i))  # Update the power of num_position_bins based on position's index

    return state_index

def state_index_to_position(state_index, num_position_bins=10, min_position=[-1.0, -1.0, 0], max_position=[1.0, 1.0, 2]):
    position = []

    for i in range(len(min_position)):
        bin_index = state_index % num_position_bins
        pos = min_position[i] + (max_position[i] - min_position[i]) * (bin_index + 0.5) / num_position_bins
        position.append(pos)
        state_index //= num_position_bins

    return position

def discretize_action(action, num_action_bins=3, min_action=[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], max_action=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
    action_index = 0

    for i in range(len(action)):
        bin_index = int(np.clip((action[i] - min_action[i]) / (max_action[i] - min_action[i]) * num_action_bins, 0, num_action_bins - 1))
        action_index += bin_index * (num_action_bins ** i)

    return action_index

def action_index_to_action(action_index, num_action_bins=3, min_action=[-1.0] * 8, max_action=[1.0] * 8):
    action = []

    for i in range(len(min_action)):
        bin_index = action_index % num_action_bins
        act = min_action[i] + (max_action[i] - min_action[i]) * (bin_index + 0.5) / num_action_bins
        action.append(act)
        action_index //= num_action_bins

    return action
