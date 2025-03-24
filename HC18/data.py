import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import os
from PIL import Image
from torchvision import transforms

df_train = pd.read_csv("./HC18/dataset/training_set_pixel_size_and_HC.csv")
df_test = pd.read_csv("./HC18/dataset/test_set_pixel_size.csv")
transform = transforms.Compose([
    transforms.Resize((540, 800)),
    # transforms.Resize((135, 200)), # Test for faster experiment
    transforms.ToTensor(),  # Converts PIL image to tensor & scales to [0,1]
])

class HC18Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        annotation_file,
        transform = None,
        # isTrain: bool = True,
    ):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform

        with open(annotation_file, 'r', encoding='utf8') as ann_file:
            lines = ann_file.readlines()[1:]
            image_files = [l.split(',')[0] for l in lines]
        seg_image_files = [f"{l.split('.')[0]}_Annotation.{l.split('.')[1]}" for l in image_files]
        self.n_samples = len(image_files)

        image_paths = []
        seg_paths = []
        
        for i in tqdm(range(self.n_samples), ncols=100):
            image_path = os.path.join(root_dir, image_files[i])
            image_paths.append(image_path)
            seg_path = os.path.join(root_dir, seg_image_files[i])
            seg_paths.append(seg_path)
            
        self.image_paths = image_paths
        self.seg_paths = seg_paths
            
    def label_fill_in(self, index):
        gray = cv2.imread(self.seg_paths[index], cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        seed_point = (10, 10)  # Surely outside the hole
        fill_color = 255  # White color to fill the hole (grayscale value)
        cv2.floodFill(gray, mask, seed_point, fill_color, loDiff=5, upDiff=5, flags=cv2.FLOODFILL_FIXED_RANGE)
        inverted = cv2.bitwise_not(gray)
    
        return inverted

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        seg = self.label_fill_in(index)
        seg = Image.fromarray(seg)
        if self.transform:
            img = self.transform(img)
            seg = self.transform(seg)
        sample = {"image": img, "segmentation": seg}
        return sample
        
    def __len__(self):
        return len(self.image_paths)
    
