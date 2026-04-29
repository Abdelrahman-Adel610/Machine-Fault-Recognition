import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=6):
    """
    Loads pre-trained ResNet18 and modifies the final layer for our specific classes.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Change the final output layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
