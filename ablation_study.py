# ablation_study.py
import torch
from torch.utils.data import DataLoader
from train_loop import evaluate
from kan_lora_model import KANLoRA
from ltm_lora_model import LTMLoRA
from utils import set_seed

def ablation_study(model, ablation_configs, data_loader, criterion, device='cuda'):
    """
    Perform an ablation study by training and evaluating models with different configurations.
    :param model: The base model.
    :param ablation_configs: List of configurations to test (e.g., removing components).
    :param data_loader: DataLoader for the evaluation data.
    :param criterion: Loss function.
    :param device: Device to perform evaluation on ('cuda' or 'cpu').
    :return: Results of the ablation study.
    """
    results = {}
    
    for config in ablation_configs:
        print(f"Running ablation with config: {config}")
        
        # Modify the model based on the configuration
        ablated_model = model
        
        # For each ablation config, we disable certain components of the model
        if config.get("remove_kan_lora", False):
            ablated_model.kan_lora = None
            print("KAN-LoRA removed")
        
        if config.get("remove_ltm_lora", False):
            ablated_model.ltm_lora = None
            print("LTM-LoRA removed")
        
        if config.get("remove_critic", False):
            ablated_model.critic = None
            print("Token-level Critic removed")
        
        # Evaluate the ablated model
        valid_loss, valid_accuracy = evaluate(ablated_model, data_loader, criterion, device)
        results[config] = {"Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy}
    
    return results

def print_ablation_results(results):
    """
    Print the results of the ablation study.
    :param results: The results dictionary from the ablation study.
    """
    for config, result in results.items():
        print(f"\nAblation config: {config}")
        print(f"Validation Loss: {result['Validation Loss']:.4f}, Validation Accuracy: {result['Validation Accuracy'] * 100:.2f}%")

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Example of how to set up an ablation study
    model = KANLoRA(input_dim=512, output_dim=256, rank=8)  # Example model
    criterion = torch.nn.CrossEntropyLoss()  # Loss function
    
    # Example ablation configurations
    ablation_configs = [
        {"remove_kan_lora": True, "remove_ltm_lora": False, "remove_critic": False},
        {"remove_kan_lora": False, "remove_ltm_lora": True, "remove_critic": False},
        {"remove_kan_lora": False, "remove_ltm_lora": False, "remove_critic": True},
        {"remove_kan_lora": True, "remove_ltm_lora": True, "remove_critic": False}
    ]
    
    # Assuming `valid_loader` is your validation DataLoader
    valid_loader = DataLoader()  # Replace with actual data loader
    
    # Run the ablation study
    results = ablation_study(model, ablation_configs, valid_loader, criterion)
    
    # Print results
    print_ablation_results(results)

if __name__ == "__main__":
    main()
