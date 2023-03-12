from keras.datasets import mnist
import numpy as np  
import matplotlib.pyplot as plt
import torch
import torchvision
from trainer import Trainer
import yaml


with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

## LOAD DATASET
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
train_dataloader = torchvision.datasets.MNIST(
    'datasets/MNIST',
    train=True,
    download=True,
    )
test_dataloader = torchvision.datasets.MNIST(
    'datasets/MNIST',
    train=False,
    download=True,
    )

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=configs['train']['batch_size'],
                                          shuffle=True,
                                          num_workers=configs['train']['num_workers'])

test_dataloader =   torch.utils.data.DataLoader(train_dataset,
                                          batch_size=configs['test']['batch_size'],
                                          shuffle=True,
                                          num_workers=configs['test']['num_workers'])                                    


## Create Trainer
trainer = Trainer(configs)

## TRAIN

#trainer.train(dataloader)

## EVALUATE

#train_losses = trainer.evaluate(train_dataloader)
#seval_losses = trainer.evaluate(test_dataloader)





