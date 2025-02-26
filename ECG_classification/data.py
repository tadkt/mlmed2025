import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


batch_size = 32


train_data = pd.read_csv("/kaggle/input/heartbeat/mitbih_train.csv")
test_data = pd.read_csv("/kaggle/input/heartbeat/mitbih_test.csv")

ecg_train = train_data.iloc[:,:-1].values
label_train = train_data.iloc[:,-1].values
ecg_test = test_data.iloc[:,:-1].values
label_test = test_data.iloc[:,-1].values

scaler = MinMaxScaler()
ecg_train_scaled = scaler.fit_transform(ecg_train)
ecg_test_scaled = scaler.transform(ecg_test)

# Define the number of samples for the validation set based on the test data size
n_test_samples = ecg_test.shape[0]

# Split the training data into new training and validation sets
ecg_val, ecg_test, label_val, label_test = train_test_split(ecg_test_scaled, label_test, test_size=0.5, random_state=6, stratify=label_test)

# Reshape the data for LSTM (samples, timesteps, features)
ecg_train_scaled = np.reshape(ecg_train_scaled, (ecg_train_scaled.shape[0], ecg_train_scaled.shape[1]))
ecg_val = np.reshape(ecg_val, (ecg_val.shape[0], ecg_val.shape[1]))
ecg_test = np.reshape(ecg_test, (ecg_test.shape[0], ecg_test.shape[1]))
# Train
ecg_train_scaled = torch.tensor(ecg_train_scaled, dtype=torch.float32)
label_train = torch.tensor(label_train, dtype=torch.long).view(-1)
train_dataset = TensorDataset(ecg_train_scaled, label_train)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# Val
ecg_val = torch.tensor(ecg_val, dtype=torch.float32)
label_val = torch.tensor(label_val, dtype=torch.long).view(-1)
val_dataset = TensorDataset(ecg_val, label_val)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)