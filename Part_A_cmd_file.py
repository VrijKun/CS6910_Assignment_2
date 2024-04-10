# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:58:12 2024

@author: ASL 7
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:43:10 2024

@author: ASL 5
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms #,datasets
import pytorch_lightning as pl

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import argparse
#from torchsummary import summary
from pytorch_lightning.callbacks import EarlyStopping
#import torch.optim as optim
import wandb
wandb.login(key='3c21150eb43b007ee446a1ff6e87f640ec7528c4')

#-----------------------------------------------Dataset processing------------------------------------

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# New Load dataset
train_data_directory = r'C:\Users\ASL 7\Downloads\DL Assignment 2\nature_12K\inaturalist_12K\train'
test_data_directory = r'C:\Users\ASL 7\Downloads\DL Assignment 2\nature_12K\inaturalist_12K\val'
    
train_dataset = ImageFolder(root = train_data_directory ,transform = transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
val_dataset = ImageFolder(root = train_data_directory, transform = transform)
test_dataset = ImageFolder(root = test_data_directory, transform = transform)


# Create data loaders
batch_size = 32
def create_data_loaders(train_data, val_data, batch_size, shuffle_train=True, pin_memory=True, num_workers=2 , persistent_workers=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train, pin_memory=pin_memory, num_workers=num_workers,  persistent_workers=persistent_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,  persistent_workers=persistent_workers)
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(train_data, val_data, batch_size = 32, shuffle_train=True)


#---------------------------------------------------Creating model---------------------------------------------------

class ECNN(pl.LightningModule):
    def __init__(self, input_shape, num_classes, num_filters, filter_size, dense_neurons, batch_normalization, dropout, activation):
        super().__init__()
        self.input_shape = input_shape
        #print(self.input_shape)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dense_neurons = dense_neurons
        self.activation = activation
        self.batch_normalization=batch_normalization

        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, num_filters[0], filter_size[0])
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], filter_size[1])
        self.conv3 = nn.Conv2d(num_filters[1], num_filters[2], filter_size[2])
        self.conv4 = nn.Conv2d(num_filters[2], num_filters[3], filter_size[3])
        self.conv5 = nn.Conv2d(num_filters[3], num_filters[4], filter_size[4])

        # Defining batch normalization
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.bn2 = nn.BatchNorm2d(num_filters[1])
        self.bn3 = nn.BatchNorm2d(num_filters[2])
        self.bn4 = nn.BatchNorm2d(num_filters[3])
        self.bn5 = nn.BatchNorm2d(num_filters[4])


        # Define pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()

        # Define dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        # Calculate the size of the feature maps after max pooling
        self.flatten_size = self.calculate_flatten_size()

        # Define dense layers
        self.fc1 = nn.Linear(self.flatten_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, num_classes)


    def forward(self, x):
        if self.batch_normalization:
          x = self.activation(self.bn1(self.conv1(x)))
        else:
          x = self.activation(self.conv1(x))
        x=self.dropout1(x)
        x = self.pool(x)
        #print('0: ',x.shape)

        if self.batch_normalization:
          x = self.activation(self.bn2(self.conv2(x)))
        else:
          x = self.activation(self.conv2(x))
        x=self.dropout2(x)
        x = self.pool(x)
        #print('1: ',x.shape)

        if self.batch_normalization:
          x = self.activation(self.bn3(self.conv3(x)))
        else:
          x = self.activation(self.conv3(x))
        x=self.dropout3(x)
        x = self.pool(x)
        #print('2: ',x.shape)

        if self.batch_normalization:
          x = self.activation(self.bn4(self.conv4(x)))
        else:
          x = self.activation(self.conv4(x))
        x=self.dropout4(x)
        x = self.pool(x)
        #print('3: ',x.shape)

        if self.batch_normalization:
          x = self.activation(self.bn5(self.conv5(x)))
        else:
          x = self.activation(self.conv5(x))
        x=self.dropout5(x)
        x = self.pool(x)



        x = x.view(x.size(0), -1)
        #print('5: ',x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print('5.5: ',x.shape)
        x = F.log_softmax(x, dim=1) # trying if it works
        
        return x

    def calculate_flatten_size(self):

        x = torch.randn(1, *self.input_shape)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        return x.view(x.size(0), -1).size(1)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        acc = acc * 100
        
        #self.log('train_loss', loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True)  # Log the training loss
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True)  # Log accuracy
        #print(f"Train Loss: {loss.item()}, train_Accuracy: {acc_percent}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y).item() / (len(y) * 1.0)
        acc = acc * 100
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_accuracy', acc, on_step=True, on_epoch=True) 
        #print(f"Validation Loss: {loss.item()}, val_Accuracy: {acc_percent}")
        return loss


 #------------------------------------running everything and running out of fuel and RAM-------------------------------------------------------------   

if __name__ == '__main__':
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
      print('Using GPU...!')
    else:
      print('Using CPU...!(terminate the runtime and restart using GPU)')
      
    
    parser = argparse.ArgumentParser(description='Train a CNN on iNaturalist Dataset...')
    parser.add_argument('--wandb_entity', type=str, default='ed23d015', help='Name of the wandb entity')
    parser.add_argument('--wandb_project', type=str, default='DL_Assignment_2', help='Name of the wandb project')
    parser.add_argument('--activation', type=str, default='elu', help='choices:[relu,gelu,elu,silu]')
    parser.add_argument('--num_filter', type=list, default=[64,128,256,512,1024], help='Enter 5 number of filters list')
    parser.add_argument('--filter_size', type=list, default=[5,5,5,5,5], help='Enter 5 filter size values')
    parser.add_argument('--dropout', type=int, default=0.4, help='choices:(0,1)')
    parser.add_argument('--batch_normal', type=bool, default=False, help='choices:["True","False"]')
    parser.add_argument('--data_augmentation', type=bool, default=False, help='choices:["True","False"]')
    parser.add_argument('--batch_size', type=int, default=32, help='choices:[32,64]')


    args = parser.parse_args()
    
    sweep_config = {
        'method': 'bayes', #grid, random,bayes
        'metric': {
          'name': 'val_accuracy',
          'goal': 'maximize'
        },
        'parameters': {
            'activation': {
                'values': [args.activation]
            },
            'num_filter': {
                'values': [args.num_filter]
            },
            'filter_size':{
                'values': [args.filter_size]
            },
            'dropout':{
                'values':[args.dropout]
            },
            'batch_normal':{
                'values':[args.batch_normal]
            },
            'batch_size':{
                'values':[args.batch_size]
            },
            'data_augmentation':{
                'values':[args.data_augmentation]
            },
            'input_shape':{
                'values':[(3, 224, 224)]
            },
            'num_classes':{
                'values': [10]
            },
        }
    }
        
    
    def sweep_train():
      # Default hyper-parameters
      config_defaults = {
          'activation':'relu',
          'num_filter':[32,32,32,32,32],
          'filter_size':[3,3,3,3,3],
          'dropout':0.3,
          'batch_normal':True,
          'batch_size': 32,
          'input_shape':(3, 224, 224),
          'num_classes': 10,
      }
    
      # Initialize wandb run
      wandb.init(project=args.wandb_project, entity=args.wandb_entity,config=config_defaults)
      wandb.run.name = 'EVALUATION RUN(ED23D015):' 'act:'+ str(wandb.config.activation)+' ;filter:'+str(wandb.config.num_filter)+ ' ;ker:'+str(wandb.config.filter_size)+ ' ;drop:'+str(wandb.config.dropout)+' ;b_n:'+str(wandb.config.batch_normal) +' ;b_s:'+str(wandb.config.batch_size)
    
      config = wandb.config
      activation = config.activation
      num_filter = config.num_filter
      filter_size = config.filter_size
      dropout = config.dropout
      batch_normal = config.batch_normal
      batch_size = config.batch_size
      input_shape = config.input_shape
      num_classes = config.num_classes
      
    
      # Loading data_loader here
      train_loader, val_loader = create_data_loaders(train_data, val_data, batch_size = batch_size , shuffle_train=True, pin_memory=True, num_workers = 2, persistent_workers=True)
      
      # implementing CNN
      model = ECNN(input_shape=input_shape, num_classes=num_classes, num_filters=num_filter, filter_size=filter_size, dense_neurons=128,batch_normalization= batch_normal, dropout=dropout ,activation=activation)
      
      
      # Initialize Lightning trainer with GPU support
      #trainer = pl.Trainer(max_epochs=2, accelerator = "gpu" if torch.cuda.is_available() else "cpu", devices = 1 if torch.cuda.is_available() else 2, log_every_n_steps=30, callbacks=[EarlyStopping(monitor='val_loss')])
      trainer = pl.Trainer(
            max_epochs= 16, 
            accelerator="gpu" if torch.cuda.is_available() else "cpu", 
            devices=1 if torch.cuda.is_available() else 2, 
            log_every_n_steps=30,
            callbacks=[EarlyStopping(monitor='val_loss')]  # Specify the strategy here didn't work strategy='ddp_spawn' and strategy='ddp_notebook'
        )
    
      for epoch in range(15):
          
          # Train for one epoch
          
          trainer.fit(model, train_loader, val_loader)
          
          if trainer.logged_metrics:  # Check if trainer.logged_metrics is not empty
                # Retrieve the metrics
                train_loss = trainer.logged_metrics.get('train_loss_epoch', float('NaN')).item()
                train_accuracy = trainer.logged_metrics.get('train_acc', float('NaN')).item()
                val_loss = trainer.logged_metrics.get('val_loss_epoch', float('NaN')).item()
                val_accuracy = trainer.logged_metrics.get('val_acc', float('NaN')).item()
            
                # Log the metrics using WandB
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                })
            
                # Print training progress
                print(f"Epoch {epoch + 1}/{trainer.max_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                if epoch==trainer.max_epochs-1:
                    torch.cuda.empty_cache()
          else:
                print("No metrics logged for this epoch.")
                if epoch==trainer.max_epochs-1:
                    torch.cuda.empty_cache()
                break  # Stop the loop if metrics are not logged
    
    
    sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
    
    wandb.agent(sweep_id, function=sweep_train, count=100) #change the number of count here 120
        
    

        
        