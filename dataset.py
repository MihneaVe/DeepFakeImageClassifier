import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Args:
            # csv_file (string): Path to the csv file with image IDs and labels
            # img_dir (string): Directory with all the images
            # transform (callable, optional): Optional transform to be applied on images
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_id = self.data_frame.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Check if there's a label column (for train/validation)
        if len(self.data_frame.columns) > 1:
            label = self.data_frame.iloc[idx, 1]
            return image, label
        else:
            # For test set without labels
            return image, img_id