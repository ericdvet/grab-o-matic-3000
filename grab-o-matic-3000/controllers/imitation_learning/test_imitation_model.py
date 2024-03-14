import numpy as np
import torch
import random
from math import *
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from model import ImitationModel, ImitationModelComplex, ImitationModelFiveLayers

numberOfTests = 100
gravity = 9.81

data_obs = np.load('observations.npz')
data_act = np.load('actions.npz')
data_out = np.load('outcomes.npz')

new_observations = data_obs['arr_0'][:100]
new_correct_actions = data_act['arr_0'][:100]
outcomes = data_out['arr_0'][:100]


# new_observations = np.array(new_observations)
# new_correct_actions = np.array(new_correct_actions)

model5 = ImitationModelFiveLayers(input_size=new_observations.shape[1], output_size=new_correct_actions.shape[1])
model5.load_state_dict(torch.load('imitation_model_five_layers.pth'))
model5.eval()

model3 = ImitationModel(input_size=new_observations.shape[1], output_size=new_correct_actions.shape[1])
model3.load_state_dict(torch.load('imitation_model.pth'))
model3.eval()

scaler = joblib.load('scaler.pkl')
new_observations_scaled = scaler.fit_transform(new_observations)

# Convert the preprocessed observations to a PyTorch tensor
new_observations_tensor = torch.tensor(new_observations_scaled, dtype=torch.float32)

# Use the trained model to predict actions based on the new observations
with torch.no_grad():
    predicted_actions5 = model5(new_observations_tensor)
    predicted_actions3 = model3(new_observations_tensor)

for i in range(1, numberOfTests):
    temp = abs(predicted_actions5[i] - new_correct_actions[i])
    
# Extract x, y, z components of predicted actions and correct actions
predicted_x_5 = predicted_actions5[:, 0]
predicted_y_5 = predicted_actions5[:, 1]
predicted_z_5 = predicted_actions5[:, 2]

predicted_x_3 = predicted_actions3[:, 0]
predicted_y_3 = predicted_actions3[:, 1]
predicted_z_3 = predicted_actions3[:, 2]


correct_x = new_correct_actions[:, 0]
correct_y = new_correct_actions[:, 1]
correct_z = new_correct_actions[:, 2]


error_x_3 = abs(correct_x - predicted_x_3.numpy())
error_y_3 = abs(correct_y - predicted_y_3.numpy())
error_z_3 = abs(correct_z - predicted_z_3.numpy())

error_x_5 = abs(correct_x - predicted_x_5.numpy())
error_y_5 = abs(correct_y - predicted_y_5.numpy())
error_z_5 = abs(correct_z - predicted_z_5.numpy())


# Create subplots
fig, ax = plt.subplots(3, 2, figsize=(10, 10))

# Plot predicted and correct x components
ax[0, 0].plot(predicted_x_5, label='Predicted X 5 Layers')
ax[0, 0].plot(predicted_x_3, label='Predicted X 3 Layers')
ax[0, 0].plot(correct_x, label='Correct X')
ax[0, 0].set_title('X Component')
ax[0, 0].legend(loc='upper right')

# Plot predicted and correct y components
ax[1, 0].plot(predicted_y_5, label='Predicted Y 5 Layers')
ax[1, 0].plot(predicted_y_3, label='Predicted Y 3 Layers')  # Fixed typo: 'Laters' to 'Layers'
ax[1, 0].plot(correct_y, label='Correct Y')
ax[1, 0].set_title('Y Component')
ax[1, 0].legend(loc='upper right')

# Plot predicted and correct z components
ax[2, 0].plot(predicted_z_5, label='Predicted Z 5 Layers')
ax[2, 0].plot(predicted_z_3, label='Predicted Z 3 Layers')
ax[2, 0].plot(correct_z, label='Correct Z')
ax[2, 0].set_title('Z Component')
ax[2, 0].legend(loc='upper right')

# X error
ax[0, 1].plot(error_x_5, label='Error X 5 Layers')
ax[0, 1].plot(error_x_3, label='Error X 3 Layers')  # Fixed label to match the context
ax[0, 1].set_title('X Error')
ax[0, 1].legend(loc='upper right')

# Y error
ax[1, 1].plot(error_y_5, label='Error Y 5 Layers')
ax[1, 1].plot(error_y_3, label='Error Y 3 Layers')
ax[1, 1].set_title('Y Error')
ax[1, 1].legend(loc='upper right')

# Z error
ax[2, 1].plot(error_z_5, label='Error Z 5 Layers')  # Assuming error_z_5 is defined, corrected from error_x_5
ax[2, 1].plot(error_z_3, label='Error Z 3 Layers')
ax[2, 1].set_title('Z Error')
ax[2, 1].legend(loc='upper right')

plt.tight_layout()
plt.show()
