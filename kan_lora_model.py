# kan_lora_model.py
import torch
import torch.nn as nn

class KANLoRA(nn.Module):
    """
    Implement the Kolmogorov-Arnold Network Low-Rank Adapter (KAN-LoRA).
    This model uses low-rank factorization to adapt large models efficiently.
    """
    def __init__(self, input_dim, output_dim, rank=8):
        """
        Initialize the KAN-LoRA adapter.
        :param input_dim: Input dimension of the model.
        :param output_dim: Output dimension of the model.
        :param rank: Rank for low-rank factorization (default is 8).
        """
        super(KANLoRA, self).__init__()
        self.rank = rank
        
        # Define the low-rank factorization matrices
        self.W1 = nn.Parameter(torch.randn(input_dim, rank))
        self.W2 = nn.Parameter(torch.randn(rank, output_dim))
        
        # Initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of the low-rank matrices.
        """
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x):
        """
        Forward pass through the KAN-LoRA layer.
        :param x: The input tensor.
        :return: The output tensor after low-rank adaptation.
        """
        # Apply low-rank factorization to the input
        return torch.matmul(torch.matmul(x, self.W1), self.W2)

    def get_weights(self):
        """
        Return the low-rank weight matrices for visualization or analysis.
        :return: The W1 and W2 matrices.
        """
        return self.W1, self.W2
