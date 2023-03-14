import numpy as np  
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from trainer import Trainer
import yaml


with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

train_dataset = torchvision.datasets.MNIST(
    'datasets/MNIST',
    train=True,
    download=True,
    transform=ToTensor(),
    )
test_dataset = torchvision.datasets.MNIST(
    'datasets/MNIST',
    train=False,
    download=True,
    transform=ToTensor(),
    )

train_dataloader = DataLoader(train_dataset,
                    batch_size=configs['train']['batch_size'],
                    shuffle=True,
                    num_workers=configs['train']['num_workers'])

test_dataloader =  DataLoader(train_dataset,
                    batch_size=configs['test']['batch_size'],
                    shuffle=True,
                    num_workers=configs['test']['num_workers'])                                    


## Create Trainer
trainer = Trainer(configs)

## TRAIN
trainer.train(train_dataloader)

## EVALUATE

#train_losses = trainer.evaluate(train_dataloader)
eval_stats = trainer.evaluate(test_dataloader)
print(f"Eval Accuracy: {eval_stats['accuracy']}")
print(f"Average Eval Loss: {np.mean(eval_stats['losses'])}")





