import numpy as np  
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class Leaf(torch.nn.Module):

    def __init__(self, params):

        self.input_dim = params['feature_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['num_classes']

        self.model = nn.Sequential([
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        ])

    def forward(self, features):

        prediction = self.model(features)

        return prediction