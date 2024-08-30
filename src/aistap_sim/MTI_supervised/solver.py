# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from tqdm import tqdm
import sys
from aistap_sim.utils.data_utils import view_real, view_complex

class SupervisedSolver():
    '''Class for training supervised network.
    '''
    def __init__(self, model, loss_fn, train_dataloader, val_dataloader, device, update_rule, num_epochs=10, print_every=100):
        '''Initialization for individual training components

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch supervised model to train
        loss_fn : torch.nn._Loss
            Loss function
        train_dataloader : torch.utils.data.DataLoader
            Torch DataLoader object to load training data and labels
        val_dataloader : torch.utils.data.DataLoader
            Torch DataLoader object to load validation data and labels
        device : torch.device
            Device object to specify cpu or gpu computation
        update_rule : torch.optim.Optimizer
            Optimizer update rule
        num_epochs : int
            Number of epochs to train
        print_every : int
            Interval (in number of batches) to update tqdm bar and save loss data
        '''
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.update_rule = update_rule
        self.num_epochs = num_epochs
        self.print_every = print_every

    def train(self):
        '''Main training loop

        Returns
        -------
        loss_history : list[tuple]
            Instantaneous loss history recorded at each print_every interval
        running_loss_history_train : list[tuple]
            Running average of training loss recorded at each epoch
        running_loss_history_val : list[tuple]
            Running average of validation loss recorded at each epoch
        '''
        num_batches = len(self.train_dataloader)
        optimizer = self.update_rule(self.model.parameters())
        loss_history = []
        running_loss_history_train = []
        running_loss_history_val = []
        pbar = tqdm(total=self.num_epochs*num_batches, file=sys.stdout)
        for t in range(self.num_epochs):
            pbar.set_description(f'Epoch {t+1}/{self.num_epochs}')
            running_loss_train = 0
            running_loss_val = 0
            self.model.train()
            for batch, (X, y, _, _) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device) # move to GPU
                pred, w = self.model(X, self.train_dataloader.dataset.noise_var)
                pred_sum = torch.sum(pred, dim=-1)
                y_sum = torch.sum(y, dim=-1)
                loss = self.loss_fn(pred_sum, w, y_sum)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                running_loss_train += loss.item() 

                if (t*num_batches + batch) % self.print_every == 0:
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(self.print_every)
                    loss_history.append((t*num_batches + batch, loss.item()))
                    
            running_loss_history_train.append((t, running_loss_train / len(self.train_dataloader)))

            self.model.eval()
            with torch.no_grad():
                for X, y, _, _ in self.val_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred, w = self.model(X, self.val_dataloader.dataset.noise_var)
                    pred_sum = torch.sum(pred, dim=-1)
                    y_sum = torch.sum(y, dim=-1)
                    loss = self.loss_fn(pred_sum, w, y_sum)
                    running_loss_val += loss.item()
            
            running_loss_history_val.append((t, running_loss_val / len(self.val_dataloader)))
        pbar.close()
                
        return loss_history, running_loss_history_train, running_loss_history_val
        

class MSELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, w, y):
        if torch.is_complex(y_pred):
            y_pred = torch.view_as_real(y_pred)

        if torch.is_complex(y):
            y = torch.view_as_real(y)
        
        return self.mse_loss(y_pred, y)