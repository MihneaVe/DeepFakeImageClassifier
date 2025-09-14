import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.multiprocessing import freeze_support

from dataset import ImageDataset
from model import ImageClassifier

def main():
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 5
    batch_size = 64

    # Transformation for test images
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create test dataset
    test_dataset = ImageDataset(csv_file='test.csv', 
                            img_dir='test', 
                            transform=test_transform)

    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)  # Set num_workers to 0

    # Load the best model
    model = ImageClassifier(num_classes=num_classes).to(device)  # Removed unnecessary parameter
    checkpoint = torch.load('models/resnet_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded with validation accuracy: {checkpoint['accuracy']:.2f}%")

    # Prediction
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            
            image_ids.extend(ids)
            predictions.extend(preds.cpu().numpy())

    # Create submission file
    submission = pd.DataFrame({
        'image_id': image_ids,
        'label': predictions
    })

    # Save the submission file
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file saved with {len(submission)} predictions.")

if __name__ == '__main__':
    freeze_support()
    main()