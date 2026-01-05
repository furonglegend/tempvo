# kan_lora_heatmap.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from kan_lora import KANLoRA

def plot_heatmap(weights, title="KAN-LoRA Heatmap"):
    """
    Generate a heatmap for visualizing the weights of the KAN-LoRA update.
    :param weights: The weight matrix from the KAN-LoRA update.
    :param title: Title of the heatmap.
    """
    # Convert weights to numpy for plotting
    weight_matrix = weights.cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weight_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Feature Index")
    plt.ylabel("Token Position")
    plt.show()

def evaluate_kan_lora(model, data_loader, device='cuda'):
    """
    Evaluate KAN-LoRA components by generating heatmaps for each layer's output.
    :param model: The trained model with KAN-LoRA.
    :param data_loader: DataLoader for input data.
    :param device: Device to run the evaluation on ('cuda' or 'cpu').
    """
    model.eval()
    with torch.no_grad():
        for input, _ in data_loader:
            input = input.to(device)
            output = model(input)
            # Extract weights from KAN-LoRA layers for visualization
            kan_lora_weights = model.kan_lora.get_weights()
            plot_heatmap(kan_lora_weights)
            break  # Generate heatmap for the first batch only

