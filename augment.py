import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.multiprocessing import freeze_support

def augment_dataset(csv_file, img_dir, output_dir, num_augmentations=5):
    """
    Generate augmented versions of each image and save them
    Args:
        csv_file: Path to the CSV file with image IDs and labels
        img_dir: Directory with all the images
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmented copies to create per image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original dataset
    data_frame = pd.read_csv(csv_file)
    
    # Define strong augmentation
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ])
    
    # For each image in the dataset
    augmented_data = []
    
    for idx in tqdm(range(len(data_frame)), desc="Augmenting images"):
        img_id = data_frame.iloc[idx, 0]
        img_path = os.path.join(img_dir, f"{img_id}.png")
        
        # Get label if available
        if len(data_frame.columns) > 1:
            label = data_frame.iloc[idx, 1]
        else:
            label = None
            
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Save original
        augmented_data.append({
            'image_id': img_id,
            'label': label
        })
        
        # Generate augmentations
        for i in range(num_augmentations):
            # Apply augmentation
            aug_image = augmentation(image)
            
            # Generate new ID and save
            aug_id = f"{img_id}_aug{i+1}"
            aug_path = os.path.join(output_dir, f"{aug_id}.png")
            aug_image.save(aug_path)
            
            # Add to dataframe
            augmented_data.append({
                'image_id': aug_id,
                'label': label
            })
    
    # Create new CSV
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(os.path.join(output_dir, 'augmented_data.csv'), index=False)
    
    print(f"Original dataset size: {len(data_frame)}")
    print(f"Augmented dataset size: {len(augmented_data)}")
    return os.path.join(output_dir, 'augmented_data.csv')

if __name__ == '__main__':
    freeze_support()
    
    # Define paths
    train_csv = 'train.csv'
    train_dir = 'train'
    output_dir = 'augmented_train'
    
    # Run augmentation
    augmented_csv = augment_dataset(train_csv, train_dir, output_dir, num_augmentations=4)
    print(f"Augmented dataset saved to: {augmented_csv}")
    print("To use this dataset for training, set the train_csv and train_dir parameters in train.py")