import numpy as np  
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tree import TreeNN

class Trainer():

    def __init__(self, configs):

        self.configs = configs
        self.device = configs['device']
        self.num_epochs = configs['train']['epochs']
        self.learning_rate = configs['train']['learning_rate']
        self.batch_size = configs['train']['batch_size']
        self.checkpoint_dir = configs['train']['checkpoint_dir']
        self.save_freq = configs['train']['save_freq']
        self.log_freq = configs['train']['log_freq']
        self.steps = 0
        self.epoch = 0

        self.model = TreeNN(self.configs)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, Y_pred, Y_true):

        return self.criterion(Y_pred, Y_true)

    def save_model(self):
        torch.save({
            'epoch' : self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            },f"{self.checkpoint_dir}checkpoint_{self.epoch}.pt")
        

    def load_model(self, filepath):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']

    def train_step(self,batch):

        # Zero Gradients
        self.optimizer.zero_grad()

        # Compute loss
        X = batch[0].reshape(self.batch_size, -1)
        Y_true = batch[1]
        Y_pred = self.model(X)
        loss = self.compute_loss(Y_pred, Y_true)

        # Backprop update
        loss.backward()
        self.optimizer.step()

        return loss


    def train(self,dataloader):

        self.model.train()
    
        for i in range(self.num_epochs):

            losses = []
            for batch in dataloader:

                loss = self.train_step(batch)
                losses.append(loss.item())

                if self.steps % self.log_freq == 0:
                    print(f"Average Epoch Loss after {self.steps} steps: {np.mean(losses)}")

                self.steps += 1

            if self.epoch % self.save_freq == 0:
                self.save_model()

            self.epoch += 1
            print(f"Epoch {self.epoch} Loss: {np.mean(losses)}")
             

    def evaluate(self, dataloader):

        self.model.eval()
        losses = []
        n_correct = 0
        for batch in dataloader:

            X = batch[0].reshape(self.batch_size, -1)
            Y_true = batch[1]
            Y_pred = self.model(X)
            
            pred_labels = torch.argmax(Y_pred,dim=1).item()
            n_correct += Y_true[Y_true == pred_labels].shape[0]
            
            loss = self.compute_loss(Y_pred, Y_true)
            losses.append(loss.item())
        

        accuracy = n_correct / len(dataloader.dataset)
        stats = {
            "accuracy" : accuracy,
            "losses": np.array(losses),
        }
        return stats



            