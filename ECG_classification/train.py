from model import LSTM_classification
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import time
from utils import calculate_accuracy
from data import train_loader, val_loader

### TRAINING VARIABLES ###
batch_size = 32
num_epochs = 20
validation_step = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LSTM_classification()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True)

def train():
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, targets).item()
        epoch_time = (time.time() - start_time) * 1000 / len(train_loader)

        train_loss = total_loss/len(train_loader)
        train_acc = total_acc/len(train_loader)
        
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        if (epoch+1) % validation_step == 0:
            
            model.eval()
            val_loss = 0
            val_acc = 0

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item()
                    val_acc += calculate_accuracy(outputs, y_val).item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - "
                f"{epoch_time:.0f}ms/step - "
                f"accuracy: {train_acc:.4f} - "
                f"loss: {train_loss:.4f} - "
                # f"val_accuracy: {val_acc:.4f} - "
                # f"val_loss: {val_loss:.4f} - "
                f"learning_rate: {current_lr:.4f}")

    print("Training complete...")
    
