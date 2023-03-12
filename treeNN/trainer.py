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
        self.checkpoint_dir - configs['train']['checkpoint_dir']
        self.save_freq = configs['train']['save_freq']
        self.log_freq = configs['train']['log_freq']

        self.epoch = 0

        self.model = TreeNN(self.configs)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, Y_pred, Y_true):

        return self.criterion(Y_pred, Y_true)

    def save_model(self):
        torch.save({
            'epoch' : self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            },f"{self.checkpoint_dir}_{self.current_epoch}.pt")
        

    def load_model(self, filepath):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def train_step(batch):

        # Zero Gradients
        self.optimizer.zero_grad()

        # Compute loss
        X = batch[0]
        Y_true = batch[1]
        Y_pred = model(X)
        loss = self.compute_loss(Y_pred, Y_true)

        # Backprop update
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

        self.model.train()
        for i in range(self.num_epochs):

            if self.current_epoch % self.save_freq == 0:
                self.save_model()

            if self.current_epoch % self.log_freq == 0:
                # ADD LOGGING
                continue 

            for batch in dataloader:
                loss = self.train_step(batch)

            self.current_epoch += 1

    def evaluate(dataloader):

        self.model.eval()
        losses = []
        for batch in dataloader:
            loss =  self.eval_step(batch)
            losses.append(loss.item())
        
        return np.array(losses)



            