import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
from datasets import get_dataset
from train import train, validate
from trainer import train_model
from test import test
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

# Set up argument parser
parser = argparse.ArgumentParser(description='Train BiasPruner')

parser.add_argument('--dataset', type=str, choices=['Fitzpatrick', 'ham', 'NIH'], default='Fitzpatrick', help='Name of the dataset to use.')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model.')
parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement before early stopping.')

args = parser.parse_args()

# Use arguments
num_classes = args.num_classes
learning_rate = args.learning_rate
num_epochs = args.num_epochs
patience = args.patience

# Load dataset
trainloader, valloader, testloader = get_dataset(args.dataset)

print('Done loading data...')

# Create ResNet18 model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_loss = float('inf')
counter = 0  # Counter to keep track of epochs without improvement

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update total loss
        total_loss += loss.item()

        # Update total and correct predictions for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for the epoch
    average_loss = total_loss / len(trainloader)
    accuracy = correct_predictions / total_samples

    # Print the results after each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in valloader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()

    average_val_loss = val_loss / len(valloader)

    # Check if validation loss has improved
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        counter = 0  # Reset counter
    else:
        counter += 1

    # Check for early stopping
    if counter >= patience:
        print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
        break

print('Done training...')

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    recall = recall_score(y_true, y_pred, average='weighted')
    return accuracy, balanced_accuracy, f1, recall

accuracy, balanced_accuracy, f1, recall = calculate_metrics(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')
