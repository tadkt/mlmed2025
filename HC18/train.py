import torch
from HC18.data import HC18Dataset, transform
from HC18.model import UNet
from HC18.criterion import DiceLoss, iou_coefficient, optimizer, scheduler
from torch.utils.data import DataLoader, random_split
from torch import nn
import json
import time

batch_size = 8
full_dataset = HC18Dataset(root_dir='./HC18/dataset/training_set', annotation_file='./HC18/dataset/training_set_pixel_size_and_HC.csv', transform=transform)
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet()
model.to(device)

validation_step = 1
def train():
    global model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Wrap model for multi-GPU

    loss_train_dict = []
    loss_val_dict = []
    iou_train_dict = []  # Store IoU during training
    iou_val_dict = []  # Store IoU during validation
    lr_dict = []

    dice_loss = DiceLoss()  # Use Dice Loss
    num_epochs = 50  # Example: Set your desired number of epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_iou_train = 0  # Track IoU for training
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch["image"].to(device), batch["segmentation"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = dice_loss(outputs, targets)  # Use Dice Loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute IoU for training
            iou = iou_coefficient(outputs, targets)
            total_iou_train += iou.item()
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx} of Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Train IoU: {iou:.4f}")

        epoch_time = (time.time() - start_time) * 1000 / len(train_loader)
        train_loss = total_loss / len(train_loader)
        avg_iou_train = total_iou_train / len(train_loader)

        # Append to tracking lists
        loss_train_dict.append(train_loss)
        iou_train_dict.append(avg_iou_train)

        if (epoch + 1) % validation_step == 0:
            model.eval()
            val_loss = 0
            total_iou_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    X_val, y_val = batch["image"].to(device), batch["segmentation"].to(device)
                    outputs = model(X_val)

                    loss = dice_loss(outputs, y_val)
                    val_loss += loss.item()
                    
                    # Compute IoU for validation
                    iou = iou_coefficient(outputs, y_val)
                    total_iou_val += iou.item()

            val_loss /= len(val_loader)
            avg_iou_val = total_iou_val / len(val_loader)

            print(f"###Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val IoU: {avg_iou_val:.4f}###")
            
            # Append to tracking lists
            loss_val_dict.append(val_loss)
            iou_val_dict.append(avg_iou_val)

            scheduler.step(1-avg_iou_val)

        current_lr = optimizer.param_groups[0]['lr']
        lr_dict.append(current_lr)

        print(f"###Epoch {epoch+1}/{num_epochs} - "
            f"{epoch_time:.0f}ms/step - "
            f"loss: {train_loss:.4f} - "
            f"train IoU: {avg_iou_train:.4f} - "
            f"learning_rate: {current_lr:.4f}###")
        torch.save(model.module.state_dict(), f"unet_epoch_{epoch+1}.pth") 

    training_results = {
        "loss_train": loss_train_dict,
        "loss_val": loss_val_dict,
        "iou_train": iou_train_dict,
        "iou_val": iou_val_dict,
        "learning_rate": lr_dict
    }
    with open('/kaggle/working/training_results', "w") as json_file:
        json.dump(training_results, json_file, indent=4)

    print("Training complete...")

if __name__ == "__main__":
    train()