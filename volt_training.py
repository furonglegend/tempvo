# volt_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from kan_lora import KANLoRA
from critic import TokenLevelCritic

class VOLTTrainer:
    def __init__(self, model, critic, train_data, device='cuda'):
        """
        Initialize the VOLT Trainer.
        :param model: The model to be adapted.
        :param critic: Token-level critic for feedback.
        :param train_data: The training data for training the critic.
        :param device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model = model
        self.critic = critic
        self.train_data = train_data
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    def train_critic(self, epochs=10):
        """
        Train the token-level critic using the provided training data.
        :param epochs: Number of training epochs for the critic.
        """
        self.critic.train()
        for epoch in range(epochs):
            for data in self.train_data:
                input, target = data
                input, target = input.to(self.device), target.to(self.device)
                
                # Forward pass
                predictions = self.critic(input)
                loss = nn.MSELoss()(predictions, target)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    def save_model(self, path):
        """
        Save the trained model to a file.
        :param path: File path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load a pre-trained model from a file.
        :param path: File path to load the model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

