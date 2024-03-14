import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Define neural network architecture
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


if __name__ == '__main__':
    MODEL_EXISTS = False

    # Load data
    data_obs = np.load('observations.npz')
    data_act = np.load('actions.npz')
    data_out = np.load('outcomes.npz')

    observations = data_obs['arr_0']
    actions = data_act['arr_0']
    outcomes = data_out['arr_0']

    # Preprocess data
    scaler = StandardScaler()
    observations_scaled = scaler.fit_transform(observations)
    joblib.dump(scaler, 'scaler.pkl') # saving for later

    label_encoder = LabelEncoder()
    outcomes_encoded = label_encoder.fit_transform(outcomes)

    # Split data into train and test sets
    obs_train, obs_test, act_train, act_test, out_train, out_test = train_test_split(
        observations_scaled, actions, outcomes_encoded, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    obs_train_tensor = torch.tensor(obs_train, dtype=torch.float32)
    obs_test_tensor = torch.tensor(obs_test, dtype=torch.float32)
    act_train_tensor = torch.tensor(act_train, dtype=torch.float32)
    act_test_tensor = torch.tensor(act_test, dtype=torch.float32)
    out_train_tensor = torch.tensor(out_train, dtype=torch.long)
    out_test_tensor = torch.tensor(out_test, dtype=torch.long)



    # Initialize model, loss function, and optimizer
    model = ImitationModel(input_size=observations_scaled.shape[1], output_size=actions.shape[1])
    if MODEL_EXISTS:
        print("Loading existing model")
        model.load_state_dict(torch.load('imitation_model.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 1000
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i in range(0, len(obs_train_tensor), batch_size):
            optimizer.zero_grad()
            outputs = model(obs_train_tensor[i:i+batch_size])
            loss = criterion(outputs, act_train_tensor[i:i+batch_size])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: ", epoch, " - Loss: ", running_loss)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(obs_test_tensor)
        test_loss = criterion(outputs, act_test_tensor)
        print(f'Test Loss: {test_loss.item()}')

    # Save the model
    torch.save(model.state_dict(), 'imitation_model.pth')
    print("Model trained and saved.")
