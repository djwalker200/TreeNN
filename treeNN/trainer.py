import numpy as np  
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tree import TreeNN

class Trainer():

    def __init__(self, configs):

        self.configs = configs
        self.num_epochs = configs['train']['epochs']
        self.learning_rate = configs['train']['learning_rate']
        self.batch_size = configs['train']['batch_size']
        self.save_freq = configs['train']['save_freq']
        self.log_freq = configs['train']['log_freq']

        self.epoch = 0

        self.model = TreeNN(self.configs)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, Y_pred, Y_true):

        return self.criterion(Y_pred, Y_true)

    def save_model(self, epoch, filepath):
        torch.save({
            'epoch' : epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            },filepath)
        

    def load_model(self, filepath):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def train_step(batch):

        X = batch[0]
        Y_true = batch[1]
        Y_pred = model(X)
        self.optimizer.zero_grad()
        loss = self.compute_loss(Y_pred, Y_true)
        loss.backward()
        self.optimizer.step()
        return loss

    def eval_step(batch):

        X = batch[0]
        Y_true = batch[1]
        Y_pred = model(X)
        loss = self.compute_loss(Y_pred, Y_true)
        return loss


    def train(dataloader):

        for i in range(self.num_epochs):
            for batch in dataloader:

                loss = self.train_step(batch)


    def evaluate(dataloader):

        losses = []
        for batch in dataloader:
            loss =  self.eval_step(batch)
            loss.append(loss.item())


            