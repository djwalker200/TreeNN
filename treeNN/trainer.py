import numpy as np  
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tree import TreeNN

class Trainer():

    def __init__(
        self,
        epochs,
        learning_rate,
        batch_size,
        save_freq,
        log_freq,
        params,
    ):

    self.epochs = epochs
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.save_freq = save_freq
    self.log_freq = log_freq
    self.params = params

    self.model = TreeNN(self.params)
    self.optimizer = torch.optim.Adam(self.model.parameters)
    self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, Y_pred, Y_true):

        return self.criterion(Y_pred, Y_true)

    def save_model(self, filepath):
        torch.save(self.model, filepath)

    def load_model(self, filepath):
        self.model = torch.load(filepath)

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

        for i in range(self.epochs):
            for batch in dataloader:

                loss = self.train_step(batch)


    def evaluate(dataloader):

        losses = []
        for batch in dataloader:
            loss =  self.eval_step(batch)
            loss.append(loss.item())


            