import numpy as np  
import torch
import torch.nn as nn
from node import Node   
from leaf import Leaf 

class TreeNN(nn.Module):

    def __init__(self, configs):
        super(TreeNN, self).__init__()

        self.configs = configs
        self.num_leaves = configs['num_leaves']
        self.num_classes = configs['num_classes']
        #self.nodes = nn.ModuleList([Node() for i in range(self.num_nodes)])
        self.delegator = Node(self.configs)
        self.leaves = nn.ModuleList([Leaf(self.configs) for i in range(self.num_leaves)])

    def forward(self, inputs):
        # inputs =  (batch size, feture_dim)
        (B,D) = inputs.shape
        features, probs = self.delegator(inputs)
        choices = torch.argmax(probs,dim=1)
        predictions = torch.zeros((B,self.num_classes))

        for i in range(B):
            idx = choices[i]
            predictions[i,:] =self.leaves[idx](features)
        
        return predictions