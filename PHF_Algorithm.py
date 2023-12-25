import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example: Linear model with input size 10

    def forward(self, x):
        return self.fc(x)

# Instantiate the model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Dummy input and target data (replace with your actual data)
input_data = torch.randn((32, 10))  # Batch size 32, input size 10
target_data = torch.randn((32, 1))  # Batch size 32, output size 1

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(input_data)
    
    # Compute the loss
    loss = criterion(predictions, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# After training, you can use the trained model for inference.
