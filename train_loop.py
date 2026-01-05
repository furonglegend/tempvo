# train_loop.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device='cuda'):
    """
    Perform one epoch of training on the model.
    :param model: The model to be trained.
    :param train_loader: DataLoader providing the training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer used for training.
    :param device: Device to perform training on ('cuda' or 'cpu').
    :return: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Iterate through the training data
    for data, targets in tqdm(train_loader, desc="Training"):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()  # Zero out gradients from the previous step
        
        outputs = model(data)  # Forward pass through the model
        
        # Calculate the loss
        loss = criterion(outputs, targets)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)
    
    # Calculate average loss and accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    
    return avg_loss, accuracy

def run_training(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20, device='cuda'):
    """
    Run the training process for multiple epochs, including evaluation.
    :param model: The model to be trained.
    :param train_loader: DataLoader for the training data.
    :param valid_loader: DataLoader for the validation data.
    :param criterion: Loss function.
    :param optimizer: Optimizer used for training.
    :param num_epochs: Number of epochs to train the model.
    :param device: Device to perform training on ('cuda' or 'cpu').
    :return: The trained model.
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train the model
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        
        # Validate the model
        valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device)
        
        print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy * 100:.2f}%")
    
    return model

def evaluate(model, data_loader, criterion, device='cuda'):
    """
    Evaluate the model on the given data loader (validation or test data).
    :param model: The model to be evaluated.
    :param data_loader: DataLoader providing the data to evaluate on.
    :param criterion: Loss function.
    :param device: Device to perform evaluation on ('cuda' or 'cpu').
    :return: The loss and accuracy on the evaluation data.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy
