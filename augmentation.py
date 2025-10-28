from typing import Dict, List, Tuple
from torchvision import transforms

def get_transforms(augmentation_params: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and validation transforms based on augmentation parameters"""
    
    train_transforms = [
        transforms.Resize((224, 224)),
    ]
    
    # Add augmentations based on parameters
    if augmentation_params['horizontal_flip']:
        train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if augmentation_params['rotation'] > 0:
        train_transforms.append(transforms.RandomRotation(augmentation_params['rotation']))
    
    if augmentation_params['brightness'] > 0:
        train_transforms.append(
            transforms.ColorJitter(brightness=augmentation_params['brightness'])
        )
    
    if augmentation_params['contrast'] > 0:
        train_transforms.append(
            transforms.ColorJitter(contrast=augmentation_params['contrast'])
        )
    
    # Add normalization
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(train_transforms), val_transforms