# NIH-Chest-Xrays-Image-Classification-with-PyTorch

This is an image multi-classification task with Chest X-ray images. The purpose of this project is to build an **End-to-End Computer Vision** process, including design, experimenting, training, and deployment of a complete machine learning project. 

All three models were trained on an M1 Pro Macbook with [MPS backend](https://pytorch.org/docs/stable/notes/mps.html) provided by PyTorch. MPS provides significantly better training performance than CPU (but not as good as CUDA). 


## Data

This repo uses [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset, which contains 112M of chest X-ray images collected by the National Institute of Health (NIH).

## Models
Currently, there are three models now have been experimented with, which are
- Resnet50
- ViT
- VGG16

## Training
Some hyperparameters:
- Learning Rate: 1e-5
- Epochs: 5
- Loss Function: Focal Loss
- Batch Size: 64


| Model | # params | training time | AUC ROC on test set | 
|-------| -------| ---- | --- |
| Resnet50 | 23M | 1h | 0.75 |  
| ViT | 80M  | 12h 31m | 0.76 | 
| VGG | 123M |# NIH-Chest-Xrays-Image-Classification-with-PyTorch