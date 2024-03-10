"""robotic manipulations"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import time as t
from time import time
import numpy as np
from math import *
from scipy.spatial.transform import Rotation as R
import random

gravity = 9.8

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

def generate_H(theta, a, d, alpha):
    return np.asmatrix([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)], 
                        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)], 
                        [0, sin(alpha), cos(alpha), d], 
                        [0, 0, 0, 1]])

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

def calculate_joint_vel(error, jacobian):
    kPt = 200
    kPa = 7.5
    # kPt = 3
    # kPa = 1
    scaled_error = np.concatenate((error[:3] * kPt, error[3:] * kPa), axis=0)
    # return np.linalg.inv(jacobian) @ scaled_error
    return scaled_error @ jacobian
    
def calculate_trajectory(x_init, y_init, z_init, x_vel, y_vel, z_vel, time_count):
    return [x_init+(x_vel*time_count) , y_init+(y_vel*time_count),  z_init+(z_vel*time_count) - ((gravity/2) * (time_count**2))]
