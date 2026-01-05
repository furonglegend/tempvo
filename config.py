# config.py
import os

class Config:
    """
    Configuration class to hold global settings for the project.
    This includes model hyperparameters, file paths, and training parameters.
    """
    def __init__(self):
        # Directories
        self.model_save_path = './models'
        self.checkpoint_path = './checkpoints'
        self.log_path = './logs'
        
        # Model parameters
        self.input_dim = 512  # Example input dimension
        self.output_dim = 256  # Example output dimension
        self.rank = 8  # Rank for the low-rank adapters
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Solver options for LTM-LoRA
        self.ode_solver = 'dopri5'

    def ensure_dirs_exist(self):
        """
        Ensure the directories for saving models, checkpoints, and logs exist.
        """
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

# Usage
config = Config()
config.ensure_dirs_exist()
