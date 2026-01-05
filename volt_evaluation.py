# volt_evaluation.py
import torch
from torch.utils.data import DataLoader
from kan_lora import KANLoRA
from critic import TokenLevelCritic

def evaluate(model, evaluation_data, critic, device='cuda'):
    """
    Evaluate the model's performance on a given evaluation dataset.
    :param model: The trained model.
    :param evaluation_data: The evaluation dataset.
    :param critic: The token-level critic.
    :param device: Device to run the evaluation on ('cuda' or 'cpu').
    """
    model.eval()
    critic.eval()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in evaluation_data:
            input, target = data
            input, target = input.to(device), target.to(device)
            
            # Get model predictions and critic feedback
            predictions = model(input)
            critic_feedback = critic(predictions)
            
            # Compute loss based on model's predictions and critic's feedback
            loss = nn.CrossEntropyLoss()(predictions, target)
            total_loss += loss.item()

            # Check if predictions are correct
            _, predicted_labels = torch.max(predictions, dim=1)
            correct_predictions += (predicted_labels == target).sum().item()
            total_samples += target.size(0)
    
    avg_loss = total_loss / len(evaluation_data)
    accuracy = correct_predictions / total_samples
    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

