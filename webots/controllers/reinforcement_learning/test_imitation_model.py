import numpy as np
import torch
import random
from math import *
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

numberOfTests = 100
gravity = 9.81

def generateTrainingData():
    maxRobotReach = 0.75
    minRobotReach = 0.5

    tof = random.uniform(1,2)
    if random.choice([True, False]):
        targetX = random.uniform(-maxRobotReach, -minRobotReach)
    else:
        targetX = random.uniform(minRobotReach, maxRobotReach)

    if random.choice([True, False]):
        targetY = random.uniform(-maxRobotReach, -minRobotReach)
    else:
        targetY = random.uniform(minRobotReach, maxRobotReach)

    targetZ = random.uniform(0.25, 1.25)
    
    angle = random.uniform(0, 2 * pi)
    height = 1
    radius = 4
    x = radius * cos(angle)
    y = radius * sin(angle)
    z = height

    velx = (targetX - x) / tof
    vely = (targetY - y) / tof
    velz = (targetZ - z + (1 / 2) * gravity * tof**2) / tof
    return ([x, y, z], [velx, vely, velz], [targetX, targetY, targetZ, 0, 0, 1, 0])

class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImitationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

new_observations = []
new_correct_actions = []
for i in range(numberOfTests):
    fruitPos, fruitVel, target = generateTrainingData()
    new_observations.append([fruitPos[0], fruitPos[1], fruitPos[2], fruitVel[0], fruitVel[1], fruitVel[2]])
    target_pov_correction = [-target[0], -target[1], target[2], target[3], target[4], target[5], target[6]]
    new_correct_actions.append(target_pov_correction)
new_observations = np.array(new_observations)
new_correct_actions = np.array(new_correct_actions)

model = ImitationModel(input_size=new_observations.shape[1], output_size=new_correct_actions.shape[1])
model.load_state_dict(torch.load('imitation_model.pth'))
model.eval()

scaler = joblib.load('scaler.pkl')
new_observations_scaled = scaler.fit_transform(new_observations)

# Convert the preprocessed observations to a PyTorch tensor
new_observations_tensor = torch.tensor(new_observations_scaled, dtype=torch.float32)

# Use the trained model to predict actions based on the new observations
with torch.no_grad():
    predicted_actions = model(new_observations_tensor)

for i in range(1, numberOfTests):
    temp = abs(predicted_actions[i] - new_correct_actions[i])
    
# Extract x, y, z components of predicted actions and correct actions
predicted_x = predicted_actions[:, 0]
predicted_y = predicted_actions[:, 1]
predicted_z = predicted_actions[:, 2]

correct_x = new_correct_actions[:, 0]
correct_y = new_correct_actions[:, 1]
correct_z = new_correct_actions[:, 2]

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot predicted and correct x components
axs[0].plot(predicted_x, label='Predicted X')
axs[0].plot(correct_x, label='Correct X')
axs[0].set_title('X Component')
axs[0].legend()

# Plot predicted and correct y components
axs[1].plot(predicted_y, label='Predicted Y')
axs[1].plot(correct_y, label='Correct Y')
axs[1].set_title('Y Component')
axs[1].legend()

# Plot predicted and correct z components
axs[2].plot(predicted_z, label='Predicted Z')
axs[2].plot(correct_z, label='Correct Z')
axs[2].set_title('Z Component')
axs[2].legend()

# Show plot
plt.tight_layout()
plt.show()
