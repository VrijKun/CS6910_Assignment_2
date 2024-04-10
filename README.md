# CS6910_Assignment_2


## Problem statement
Implimentation of CNNs: train from scratch and finetune a pre-trained model as it is.

##Prerequisites
```
wget
python 3.12
pytorch and pytoch-lightning 2.2.1
numpy 125
```
- I have conducted most some of my experments on Google collab. And some on PC.
- To run the code clone the repository, install wandb and wget which is used to import the dataset.
- You need to use inbuilt GPU to run the experiments on Collab. It will allow faster computation and training of your model.
```
!pip install wget
pip install --upgrade pytorch-lightning
!pip install --upgrade wandb
!pip install timm #(for EfficinetNetV2)
```
- You can run the python code locally as well. for this install using following code
```
  pip install wandb
  pip install numpy
  pip install pytorch
  pip install lightning
  pip install timm(for EfficinetNetV2)
```
- Just an FYI you need to have a good GPU setup to run this on your PC.
## Dataset used for Experiments
- [inaturalist](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)

##Part A
Followings are the Hyperparameters used in this part
| S. No.        | Hyperparameters       |  Values/types                                                    |
| ------------- | ----------------------|----------------------------- |
|1.             | Activation functions  | ReLu,GeLU,SiLu, Elu          |
|2.             | Num_filters           | [32,32,32,32,32] ,[64,64,64,64,64] ,[128,128,128,128,128],[64,128,256,512,1024] |
|3.             | Filter_size           | [3,3,3,5,5],[5,5,5,5,5],[3,3,3,3,3] |
|4.             | Dropout               | 0.2,0.3            | 
|5.             | Batch_normal          | True,False         |
|6.             | batch_size            | 32,64              |

- I have used Adam optimizer.
### Part A code
- [here](https://github.com/VrijKun/CS6910_Assignment_2/blob/5db00e3509e650262ba3e82736fb51e9797c33b5/Assignment_2_DL_ED23D015.ipynb)
- python file [here](https://github.com/VrijKun/CS6910_Assignment_2/blob/758847f9a5ace52c661ee6875ea57cf88874e930/Part_A_.py)
# Part B
### Hyperparameters used in experiments for Part B are following:
|Sr. no| Hyperparameter| Variation/values used|
|------|---------------|-----------------|
|1.| Freeze percent| 0.25, 0.55, 0.8|
|2.| Learning rate| 0.0001,0.0003|
|3.| Beta1| 0.9,0.93,0.95|

### Code for Part B

The experiments done for Part B can be found [here](https://github.com/VrijKun/CS6910_Assignment_2/blob/758847f9a5ace52c661ee6875ea57cf88874e930/DL_Assinment_2_partB.ipynb).


## Evaluation file for part A(PartA_cmd.py)

For evaluating model download [PartA_cmd.py](https://github.com/Shreyash007/CS6910-Assignment2/blob/main/PartA_cmd.py) file. (make sure you have all the prerequisite libraries installed).
Make sure you have downloaded the dataset and the directory of dataset should be changed in line 53,54
```data_dir = r'C:\Users\SHREYASH\Desktop\CS6910\Assignment 2\Dataset\nature_12K\inaturalist_12K\train'```
```data_dir_2 = r'C:\Users\SHREYASH\Desktop\CS6910\Assignment 2\Dataset\nature_12K\inaturalist_12K\val'```
To the directory location of your stored dataset

And run the following command in the command line(this will take the default arguments).
```
python PartA_cmd.py
```
The default evaluation run can be seen [here](https://wandb.ai/shreyashgadgil007/shreyashgadgil007/runs/) in wandb.


The arguments supported by PartA_cmd.py file are:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | "CS6910-Assignment2" | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | "shreyashgadgil007"  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `--num_filter` |[64,128,256,512,1024] | Enter 5 filters list |
| `--activation` | 'elu' | choices:[relu,gelu,elu,silu] |
| `--kernel_size` | [5,5,5,5,5] | Enter 5 kernel values |
| `--dropout_rate` | 0.4 | choice in range (0,1) |
| `--batch_norm` | False | True/False |
| `--data_augmentation` | False | True/False |

Supported arguments can also be found by:
```
python PartA_cmd.py -h
```
## Evaluation file for part B([PartB_cmd.py](https://github.com/Shreyash007/CS6910-Assignment2/blob/main/PartB_cmd.py))
Please follow the instructions of Part A, for loading dataset.
And run the following command in the command line(this will take the default arguments).
```
python PartB_cmd.py
```
The default evaluation run can be seen [here](https://wandb.ai/shreyashgadgil007/shreyashgadgil007/runs/) in wandb.


The arguments supported by PartB_cmd.py file are:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | "CS6910-Assignment2" | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | "shreyashgadgil007"  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `--freeze_percent` | 0.25 | choice in range (0,1) |
| `--learning_rate` | 0.0003 | choice in range (0,1)|
| `--beta1` | 0.93 | choice in range (0,1) |

## Report

The wandb report for this assignment can be found [here](https://wandb.ai/shreyashgadgil007/CS6910-Assignment2/reports/CS6910-Assignment-2--VmlldzozOTAzMTc4).
## Author
[Shreyash Gadgil](https://github.com/Shreyash007)
ED22S016

