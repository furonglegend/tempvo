# utils.py
import torch
import numpy as np

def set_seed(seed):
    """
    Set random seeds for reproducibility.
    :param seed: The seed value to be set for all random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def save_checkpoint(model, optimizer, epoch, filename):
    """
    Save the model checkpoint including the model state, optimizer state, and the current epoch.
    :param model: The model to be saved.
    :param optimizer: The optimizer used for training.
    :param epoch: The current epoch during training.
    :param filename: The file path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")

def load_checkpoint(model, optimizer, filename):
    """
    Load a model checkpoint to resume training or for evaluation.
    :param model: The model to load the checkpoint into.
    :param optimizer: The optimizer to load the checkpoint into.
    :param filename: The file path from where to load the checkpoint.
    :return: The epoch number at which the checkpoint was saved.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filename}, starting from epoch {epoch}")
    return epoch
