import time as t
from time import time

from controller import Robot
from controller import Supervisor
from controller import Keyboard

import numpy as np
from math import *
from scipy.spatial.transform import Rotation as R
import random

#Define a table of fruit velocities to apply (x,y,z)
launchTable = [[0,0,8]]
maxFruit = 1
maxRobotReach = 1
ORIGIN = np.array([0,0,0.6])
gravity = 9.81

### WRITE FUNCTION TO SET FRUIT POSITION GIVEN ANGLE AND RADIUS FROM BASE OF THE ROBOT - SET TO GIVEN HEIGHT
def genFruitPos():
    fruitNode = supervisor.getFromDef('fruit0')
    translation_field = fruitNode.getField('translation')

    angle = random.uniform(0, 2 * pi)
    height = random.uniform(1, 3)
    radius = 4
    x = radius * cos(angle)
    y = radius * sin(angle)
    
    startingPosition = [x, y, height]
    translation_field.setSFVec3f(startingPosition)
    return startingPosition

### WRITE FUNCITON TO DETERMINE A RANDOM POINT ON SIDE THAT THE FRUIT IS PLACED
### WRITE FUNCITON TO DETERMINE NEEDED VELOCITIES TO GO FROM POINT OF FRUIT TO POINT OF ROBOT
def launchFruit():
    tof = random.uniform(1,2)
    fruitNode = supervisor.getFromDef('fruit0')
    targetX, targetY, targetZ = random.uniform(-maxRobotReach, maxRobotReach), random.uniform(-maxRobotReach, maxRobotReach), random.uniform(0.25, 1.25)
    translation_field = fruitNode.getField('translation')
    x, y, z = translation_field.getSFVec3f()
    velx = (targetX - x) / tof
    vely = (targetY - y) / tof
    velz = (targetZ - z + (1 / 2 * gravity * tof**2)) / tof
    fruitNode.setVelocity([velx, vely, velz, 0, 0, 0])
    fruitNode = None
    return ([velx, vely, velz], [targetX, targetY, targetZ])

def isTouched(caught):
    prevCaught = np.copy(caught)
    for fruitIndex in range(maxFruit):
        fruitNode = supervisor.getFromDef('fruit' + str(fruitIndex))
        trans_field = fruitNode.getField("translation")
        robotPos = [x_ee, y_ee, z_ee]
        diff = np.zeros(3)
        for i in range(3):
            diff[i] = trans_field.getSFVec3f()[i] - robotPos[i]
        if np.linalg.norm(diff) < 1.25:
            return True
            caught[fruitIndex] = 1
        else:
            return False
    if(not np.array_equal(prevCaught, caught)):
        print(caught)