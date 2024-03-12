import time as t
from time import time
from controller import Robot
from controller import Supervisor
from controller import Node
from controller import Keyboard
from controller import Lidar
from controller import Camera
from controller.camera import CameraRecognitionObject
import numpy as np
from math import *
from scipy.spatial.transform import Rotation as R
import random
import torch
import torch.nn as nn
import joblib


robot = Supervisor()

timestep = 64

# Initialize camera
c = robot.getDevice("camera")
c.enable(timestep)
c.recognitionEnable(timestep)
while robot.step(timestep) != -1:
    balls = c.getRecognitionObjects()
    for ball in balls:    
        print("x: {}, y: {}".format(ball.getPositionOnImage()[0], ball.getPositionOnImage()[1]))