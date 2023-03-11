import numpy as np  
import matplotlib.pyplot as plt
import torch
from trainer import Trainer
import yaml

with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

## LOAD DATASET

## Create Trainer
trainer = Trainer(configs)

## TRAIN

## EVALUATE



