import torch

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)  # Get predicted class
    correct = (preds == labels).float().sum()
    return correct / labels.size(0)