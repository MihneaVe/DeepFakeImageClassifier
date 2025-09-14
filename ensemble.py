import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.multiprocessing import freeze_support

from dataset import ImageDataset
from model import ImageClassifier
from mixup import mixup_data, mixup_criterion

def train_ensemble_models(device):
    """Train optimized ensemble models"""
    # Simplified, proven configurations
    model_configs = [
        {'architecture': 'resnet18', 'batch_size': 32, 'learning_rate': 0.002, 'weight_decay': 1e-4},
        {'architecture': 'resnet18', 'batch_size': 24, 'learning_rate': 0.001, 'weight_decay': 2e-4},
        {'architecture': 'resnet34', 'batch_size': 16, 'learning_rate': 0.001, 'weight_decay': 1e-4},
    ]
    
    os.makedirs('models/ensemble', exist_ok=True)
    
    for i, config in enumerate(model_configs):
        print(f"\n=== Training model {i+1}/{len(model_configs)}: {config['architecture']} ===\n")
        
        import torch.nn as nn
        import torch.optim as optim
        import matplotlib.pyplot as plt
        
        # Conservative data augmentation
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
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
        
        # Create dataloaders
        batch_size = config['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=0)
        
        # Initialize model
        num_classes = 5
        model = ImageClassifier(num_classes=num_classes, 
                               architecture=config['architecture']).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()  # No label smoothing for ensemble diversity
        learning_rate = config['learning_rate']
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                               weight_decay=config['weight_decay'])
        
        # Scheduler
        num_epochs = 60
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Lists to store metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        patience = 12
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for images, labels in train_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Use mixup only 30% of the time for diversity
                if np.random.random() < 0.3:
                    mixed_images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
                    optimizer.zero_grad()
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    
                    # Approximate accuracy calculation for mixup
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (lam * (predicted == targets_a).sum().item() + 
                               (1 - lam) * (predicted == targets_b).sum().item())
                else:
                    # Standard training
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                loss.backward()
                optimizer.step()
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
            
            scheduler.step()
            
            # Save best model
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_val_loss,
                    'accuracy': accuracy,
                    'architecture': config['architecture']
                }, f'models/ensemble/model_{i+1}_{config["architecture"]}_best.pth')
                print(f"Model saved with accuracy: {accuracy:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Plotting
        plt.style.use('seaborn-v0_8')
        
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{config["architecture"]} - Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'models/ensemble/loss_plot_{i+1}_{config["architecture"]}.png')
        plt.close()
        
        plt.figure(figsize=(12, 5))
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{config["architecture"]} - Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'models/ensemble/accuracy_plot_{i+1}_{config["architecture"]}.png')
        plt.close()

def ensemble_predict(device, test_csv='test.csv', test_dir='test'):
    """Make predictions using ensemble with weighted soft voting"""
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageDataset(csv_file=test_csv, img_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load all models
    model_paths = [f for f in os.listdir('models/ensemble') if f.endswith('.pth')]
    
    if not model_paths:
        raise ValueError("No model files found!")
    
    print(f"Found {len(model_paths)} models for ensemble")
    
    models = []
    total_accuracy = 0
    
    for model_path in model_paths:
        checkpoint = torch.load(os.path.join('models/ensemble', model_path), map_location=device)
        architecture = checkpoint.get('architecture', 'resnet18')
        model = ImageClassifier(num_classes=5, architecture=architecture).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        accuracy = checkpoint.get('accuracy', 50)  # Default if missing
        total_accuracy += accuracy
        models.append((model, accuracy))
        print(f"Loaded {architecture} with accuracy: {accuracy:.2f}%")
    
    # Make predictions
    all_image_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc='Ensemble prediction'):
            images = images.to(device)
            batch_size = images.size(0)
            weighted_probs = torch.zeros((batch_size, 5), device=device)
            
            for model, accuracy in models:
                weight = accuracy / total_accuracy
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                weighted_probs += weight * probs
            
            _, preds = torch.max(weighted_probs, dim=1)
            
            all_image_ids.extend(img_ids)
            all_predictions.extend(preds.cpu().numpy())
    
    # Create submission
    submission = pd.DataFrame({
        'image_id': all_image_ids,
        'label': all_predictions
    })
    
    submission.to_csv('submission_ensemble.csv', index=False)
    print(f"Ensemble submission saved with {len(submission)} predictions.")

if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        ensemble_predict(device)
    else:
        train_ensemble_models(device)