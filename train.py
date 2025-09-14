import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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

    # Optimized hyperparameters
    num_classes = 5
    num_epochs = 80  # Reduced from 150
    batch_size = 32
    learning_rate = 0.0015  # Slightly higher

    # Moderate data augmentation - proven to work
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # Less aggressive
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Reduced rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Gentler
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImageDataset(csv_file='train.csv', 
                                img_dir='train', 
                                transform=train_transform)

    val_dataset = ImageDataset(csv_file='validation.csv', 
                            img_dir='validation', 
                            transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)

    # Initialize the model with ResNet18 for speed
    model = ImageClassifier(num_classes=num_classes, architecture='resnet18').to(device)

    # Optimized loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Model saving directory
    os.makedirs('models', exist_ok=True)
    
    best_val_acc = 0
    patience = 15
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Train Acc: {train_accuracy:.2f}%, '
            f'Val Loss: {epoch_val_loss:.4f}, '
            f'Val Acc: {accuracy:.2f}%')
        
        # Update scheduler
        scheduler.step()
        
        # Save the best model
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': accuracy
            }, f'models/resnet_best.pth')
            print(f"Best model saved with accuracy: {accuracy:.2f}%!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Use seaborn style for plots
    plt.style.use('seaborn-v0_8')
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/loss_plot.png')

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/accuracy_plot.png')

    print(f'Training complete! Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    freeze_support()
    main()