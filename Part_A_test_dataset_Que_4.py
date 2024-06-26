# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:17:24 2024

@author: ASL 7
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:43:10 2024

@author: ASL 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms #,datasets
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np


#from torchsummary import summary
from pytorch_lightning.callbacks import EarlyStopping
#import torch.optim as optim
import wandb
wandb.login(key='3c21150eb43b007ee446a1ff6e87f640ec7528c4')

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
          x = F.relu(self.bn1(self.conv1(x)))
        else:
          x = F.relu(self.conv1(x))
        x=self.dropout1(x)
        x = self.pool(x)
        #print('0: ',x.shape)

        if self.batch_normalization:
          x = F.relu(self.bn2(self.conv2(x)))
        else:
          x = F.relu(self.conv2(x))
        x=self.dropout2(x)
        x = self.pool(x)
        #print('1: ',x.shape)

        if self.batch_normalization:
          x = F.relu(self.bn3(self.conv3(x)))
        else:
          x = F.relu(self.conv3(x))
        x=self.dropout3(x)
        x = self.pool(x)
        #print('2: ',x.shape)

        if self.batch_normalization:
          x = F.relu(self.bn4(self.conv4(x)))
        else:
          x = F.relu(self.conv4(x))
        x=self.dropout4(x)
        x = self.pool(x)
        #print('3: ',x.shape)

        if self.batch_normalization:
          x = F.relu(self.bn5(self.conv5(x)))
        else:
          x = F.relu(self.conv5(x))
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
    '''
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('test_loss', loss)
    
    #break here

    
    
    def training_step(self, train_loader):
        self.train()
        for x, y in train_loader:
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y)
            self.log('train_loss', loss)
            return loss

    def validation_step(self, val_loader):
        self.eval()
        for x, y in val_loader:
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y)
            self.log('val_loss', loss)

    def test_step(self, test_loader):
        self.eval()
        for x, y in test_loader:
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y)
            self.log('test_loss', loss)
            #2nd break here
            '''
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
    '''
    
    def validation_accuracy(model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                y_hat = model(x)
                correct += (torch.argmax(y_hat, dim=1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        return acc

    def test_accuracy(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                y_hat = model(x)
                correct += (torch.argmax(y_hat, dim=1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        return acc

    '''

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# New Load dataset
train_data_directory = r'C:\Users\ASL 7\Downloads\DL Assignment 2\nature_12K\inaturalist_12K\train'
test_data_directory = r'C:\Users\ASL 7\Downloads\DL Assignment 2\nature_12K\inaturalist_12K\val'
    
train_dataset = ImageFolder(root = train_data_directory ,transform=transform)



#train_dataset = ImageFolder('/content/inaturalist_12K/train', transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])
#test_data = ImageFolder('/content/inaturalist_12K/val', transform=transform)
test_dataset = ImageFolder(root = test_data_directory, transform = transform)



# Create data loaders

def create_data_loaders(train_data, val_data, batch_size, shuffle_train=True, pin_memory=True, num_workers=2 , persistent_workers=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train, pin_memory=pin_memory, num_workers=num_workers,  persistent_workers=persistent_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,  persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,  persistent_workers=persistent_workers)
    return train_loader, val_loader, test_loader



# Followings are my best parameters

sweep_config = {
    'method': 'bayes', #grid, random,bayes
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'activation': {
            'values': ['relu']
        },
        'num_filter': {
            'values': [[64,128,256,512,1024]]
        },
        'filter_size':{
            'values':[[3,3,3,5,5]]
        },
        'dropout':{
            'values':[0.2]
        },
        'batch_normal':{
            'values':[False]
        },
        'batch_size':{
            'values':[64]
        },
        'data_augmentation':{
            'values':[False]
        },
        'input_shape':{
            'values':[(3, 224, 224)]
        },
        'num_classes':{
            'values': [10]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, entity='ed23d015', project="DL_Assignment_2")


def sweep_train():
  # Default hyper-parameters
  config_defaults = {
      'activation':'relu',
      'num_filter':[64,128,256,512,1024],
      'filter_size':[3,3,3,5,5],
      'dropout':0.2,
      'batch_normal':False,
      'batch_size': 64,
      'input_shape':(3, 224, 224),
      'num_classes': 10,
  }

  # Initialize wandb run
  wandb.init(project='DL_Assignment_2', entity='ed23d015',config=config_defaults)
  wandb.run.name = 'act:'+ str(wandb.config.activation)+' ;filter:'+str(wandb.config.num_filter)+ ' ;ker:'+str(wandb.config.filter_size)+ ' ;drop:'+str(wandb.config.dropout)+' ;b_n:'+str(wandb.config.batch_normal) +' ;b_s:'+str(wandb.config.batch_size)

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
  train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, batch_size = batch_size , shuffle_train=True, pin_memory=True, num_workers = 2, persistent_workers=True)
  
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
      else:
            print("No metrics logged for this epoch.")
            break  # Stop the loop if metrics are not logged

    
  model.eval()
    

  # Iterate over the test dataset
  test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
  images, labels = next(iter(test_loader))
    
  # Make predictions using the model
  with torch.no_grad():
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Define class labels (replace with your actual class labels)
  class_labels = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
    
    # Create a grid to display images and predictions
  fig, axes = plt.subplots(10, 4, figsize=(15, 30))
    
  for i, ax in enumerate(axes):
        # Display sample image
        ax[0].imshow(np.transpose(images[i], (1, 2, 0)))
        ax[0].axis('off')
        
        # Display true label
        true_label = class_labels[labels[i]]
        ax[1].text(0.5, 0.5, f'True Label: {true_label}', fontsize=12, ha='center', color='green')
        ax[1].axis('off')
        
        # Display predicted label with confidence score
        predicted_label = class_labels[predicted[i]]
        confidence_score = probabilities[i][predicted[i]].item()
        ax[2].text(0.5, 0.5, f'Predicted: {predicted_label} ({confidence_score:.2f})', fontsize=12, ha='center', color='red' if true_label != predicted_label else 'green')
        ax[2].axis('off')
        
        # Highlight correct and incorrect predictions
        if true_label != predicted_label:
            ax[3].text(0.5, 0.5, 'Incorrect Prediction', fontsize=12, ha='center', color='red')
        else:
            ax[3].text(0.5, 0.5, 'Correct Prediction', fontsize=12, ha='center', color='green')
        ax[3].axis('off')
    
  plt.tight_layout()
    
    # Save the figure as an artifact in wandb
  wandb.log({"test_predictions": plt}, commit=False)
        
      

if __name__ == "__main__":
    wandb.agent(sweep_id, function=sweep_train, count = 1) #change the number of count here 120
    
    
    