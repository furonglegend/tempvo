# volt_inference.py
import torch
from kan_lora import KANLoRA
from critic import TokenLevelCritic

def infer(model, input_data, critic, device='cuda'):
    """
    Run inference with the trained model and token-level critic.
    :param model: The trained model.
    :param input_data: Input data for inference.
    :param critic: Token-level critic to guide the updates.
    :param device: Device to run inference on ('cuda' or 'cpu').
    """
    model.eval()
    critic.eval()
    
    input_data = input_data.to(device)
    
    with torch.no_grad():
        # Generate output using the trained model
        output = model(input_data)
        
        # Get feedback from the critic for each token in the sequence
        critic_feedback = critic(output)
        
        # Apply the critic feedback to adjust the model's predictions (if applicable)
        adjusted_output = output + critic_feedback
        
        return adjusted_output

def save_inference_output(output, path):
    """
    Save the inference output to a file.
    :param output: The output from the inference process.
    :param path: The path to save the output.
    """
    torch.save(output, path)
    print(f"Inference output saved to {path}")
