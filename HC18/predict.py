from PIL import Image
from HC18.data import transform
from HC18.model import UNet
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet()
model_path = "./HC18/unet_epoch_50.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image

# Inference function
def predict(image_path):
    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image)

    output = output.squeeze().cpu().numpy()  # Convert to numpy array
    return (output > 0.5).astype(np.uint8)  # Apply threshold (binarize)

test_image_path = "./HC18/dataset/test_set/000_HC.png"
pred_mask = predict(test_image_path)

# Display result
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(Image.open(test_image_path), cmap="gray")
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.show()