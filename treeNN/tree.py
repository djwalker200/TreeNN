import numpy as np  
import torch
import matplotlib.pyplot as plt

from node import Node   
from leaf import Leaf 
class TreeNNs(torch.nn.Module):

    def __init__(self, params):

        self.params = params
        self.num_leaves = params['num_leaves']
        self.output_dim = params['num_classes']
        #self.nodes = nn.ModuleList([Node() for i in range(self.num_nodes)])
        self.delegator = Node(self.params)
        self.leaves = nn.ModuleList([Leaf(self.params) for i in range(self.num_leaves)])

    def forward(self, inputs):
        # inputs =  (batch size, feture_dim)
        (B,D) = inputs.shape
        features, probs = self.delegator(x)
        choices = torch.argmax(probs,dim=1)
        predictions = torch.zeros((B,self.num_classes))

        for i in range(B):
            idx = choices[i]
            predictions[i,:] =self.leaves[idx](features)
        
        return predictions