# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:58:22 2024

@author: ASL 7
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import timm
import wandb

# Logging into wandb
wandb.login(key='3c21150eb43b007ee446a1ff6e87f640ec7528c4')

# Preparing data for training and validation

def data_processing(data_augmentation=True):
  if data_augmentation:
    t_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Above values are standard values used in imagenet competition
    ])
  else:
        t_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Above values are standard values used in imagenet competition
    ])
  
  

  # Create training and validation datasets
  train_dataset = ImageFolder(root=r'C:\Users\ASL 7\Downloads\DL Assignment 2\nature_12K\inaturalist_12K\train', transform=t_transform)
  train_size = int(0.8 * len(train_dataset))
  valid_size = len(train_dataset) - train_size
  train_data, val_data = random_split(train_dataset, [train_size, valid_size])


  # Create training and validation data loaders
  BATCH_SIZE=32
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  valid_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
  return train_loader,valid_loader
  # Create training and validation data loaders

  train_loader,valid_loader=data_processing()

# defining efficientnet here, will be used in wandb sweep later
def efficientnet(freeze_percent):
    # Load the EfficientNetV2 model with pre-trained weights
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
    
    # Freeze a percentage of layers
    count_total = sum(1 for _ in model.parameters())
    count = 0
    for param in model.parameters():
        if count < int(freeze_percent * count_total):
            param.requires_grad = False
            count += 1
        else:
            break
    
    # Replace the last layer with a new one for the specified number of classes
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 10)
    
    return model


sweep_config = {
    'method': 'grid', #grid, random,bayes
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'  
    },
    'parameters': {
        'freeze_percent': {
            'values': [0.8] # 0.25, 0.55, 0.8
        },        
        'learning_rate':{
            'values':[0.0001,0.0003]
        },
        'beta1':{
            'values':[0.9,0.93,0.95]
        }
        
    }
}

sweep_id = wandb.sweep(sweep_config, entity='ed23d015', project="DL_Assignment_2")

def sweep_train():
  # Default values for hyper-parameters we're going to sweep over
  config_defaults = {
      'freeze_percent':0.25,
      'learning_rate':0.0003,
      'beta1':0.93,
  }

  # Initialize a new wandb run
  wandb.init(project='DL_Assignment_2', entity='ed23d015',config=config_defaults)
  wandb.run.name = 'fp:'+ str(wandb.config.freeze_percent)+' ;lr:'+str(wandb.config.learning_rate)+ ' ;beta1:'+str(wandb.config.beta1)

  
  config = wandb.config
  learning_rate = config.learning_rate
  freeze_percent = config.freeze_percent
  beta1 = config.beta1
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Starting and checking GPU availability
  
  # model training here
  model = efficientnet(freeze_percent).to(device)
  train_loader,valid_loader = data_processing()
  criterion = nn.CrossEntropyLoss()
  
  optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(beta1,0.95))#using adam optimizer
  
  num_epochs = 10
  for epoch in range(num_epochs):
          # Set to training mode
          model.train()

          loss_run = 0.0 # Running loss in each epoch
          run_corr = 0
          for inputs, labels in train_loader:
              inputs, labels = inputs.to(device), labels.to(device)

              # Zero the parameter gradients
              optimizer.zero_grad()

              # Forward pass
              otps = model(inputs)
              _, preds = torch.max(otps, 1)
              loss = criterion(otps, labels)
              
              # Applying Backward pass
              loss.backward()
              
              # Optimizing
              optimizer.step()

              # loss and corrects
              loss_run += loss.item() * inputs.size(0)
              run_corr += torch.sum(preds == labels.data)

          train_epoch_loss = loss_run / len(train_loader.dataset)
          train_epoch_acc = (run_corr.double() / len(train_loader.dataset))*100

          # Doing evaluation on validation set
          model.eval()

          loss_run = 0.0
          run_corr = 0
          with torch.no_grad():
              for inputs, labels in valid_loader:
                  inputs, labels = inputs.to(device), labels.to(device)

                  # Forward pass
                  otps = model(inputs)
                  _, preds = torch.max(otps, 1)
                  loss = criterion(otps, labels)

                  # Statistics
                  loss_run += loss.item() * inputs.size(0)
                  run_corr += torch.sum(preds == labels.data)

          valid_epoch_loss = loss_run / len(valid_loader.dataset)
          valid_epoch_acc = (run_corr.double() / len(valid_loader.dataset))*100
          
          # Tracking the progress of epoch here
          print(f"Epoch {epoch+1}/{num_epochs}--> Training_Loss:{train_epoch_loss:.2f}; Train_Accuracy:{train_epoch_acc:.2f}; Validation_Loss:{valid_epoch_loss:.2f}; Val_Accuracy:{valid_epoch_acc:.2f}")
          wandb.log({"train_loss":train_epoch_loss,"train_accuracy": train_epoch_acc,"val_loss":valid_epoch_loss,"val_accuracy":valid_epoch_acc},)
          
          if epoch==num_epochs-1:
            torch.cuda.empty_cache() # making some space here


if __name__ == '__main__':
    wandb.agent(sweep_id, function=sweep_train, count = 3) 