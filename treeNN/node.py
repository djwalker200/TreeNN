import numpy as np  
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Node(torch.nn.Module):

    def __init__(self, params):
        
        self.input_dim = params['input_dim'] 
        self.hidden_dim = params["hidden_dim"]
        self.feature_dim = params['feature_dim']
        self.num_leaves = params["num_leaves"]

        # Encode input to hidden vector
        self.encoder = nn.Sequential([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ])

        # Map hidden vector to leaf prediction
        self.predictor = nn.Linear(self.hidden_dim, self.num_leaves)

        # Create conditioned feature vector
        self.feature_encoder = nn.Linear(self.hidden_dim + self.num_leaves, self.feature_dim)

    def forward(self, inputs):
        
        # Get hidden vector
        hidden = self.encoder(inputs)

        # Get leaf prediction
        probs = self.predictor(hidden)

        # Get conditioned feature vector
        features = self.feature_encoder(torch.cat((hidden,probs), dim=1))

        return features, probs

    

