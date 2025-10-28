import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Dict, List, Tuple
from augmentation import get_transforms
from Dataset import PneumoniaDataset
from CNN_architecture import PneumoniaCNN
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # Calculate additional metrics
    tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return epoch_loss, epoch_acc, precision, recall, f1


def train_model(hyperparams: Dict, data_path: str, epochs: int = 10):
    """Train model with given hyperparameters"""
    
    # Get transforms
    train_transform, val_transform = get_transforms(hyperparams['augmentation'])
    
    # Create datasets
    train_dataset = PneumoniaDataset(data_path, 'train', transform=train_transform)
    val_dataset = PneumoniaDataset(data_path, 'val', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], 
                          shuffle=False, num_workers=2)
    
    # Create model
    model = PneumoniaCNN(
        num_filters=hyperparams['num_filters'],
        dropout_rate=hyperparams['dropout_rate']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'],
                          weight_decay=hyperparams['weight_decay'])
    
    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1 = evaluate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, F1: {f1:.4f}")
    
    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    return model, best_val_acc, f1