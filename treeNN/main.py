from keras.datasets import mnist
import numpy as np  
import matplotlib.pyplot as plt
import torch
from trainer import Trainer
import yaml


with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

## LOAD DATASET
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
train_dataloader = DataLoader(train_X, train_y)
test_dataloader = DataLoader(test_X, test_y)
## Create Trainer
trainer = Trainer(configs)

## TRAIN

#trainer.train(dataloader)

## EVALUATE

train_losses = trainer.evaluate(train_dataloader)
eval_losses = trainer.evaluate(test_dataloader)





