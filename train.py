import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from datetime import datetime

from models.coral_reef_detector import CoralReefDetector
from models.fish_detector import FishDetector
from models.illegal_fishing_detector import IllegalFishingDetector
from data.dataloader import MarineDataset, create_data_loaders

def train_coral_reef_model(train_dir, val_dir, output_dir, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CoralReefDetector(num_classes=4)
    model.to(device)
    
    train_loader, val_loader = create_data_loaders(train_dir, val_dir)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        scheduler.step()
        
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy
        }
        training_history.append(epoch_info)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(output_dir) / 'best_coral_reef_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved: {model_path}')
    
    final_model_path = Path(output_dir) / 'final_coral_reef_model.pth'
    torch.save(model.state_dict(), final_model_path)
    
    history_path = Path(output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f'Training completed. Final model saved: {final_model_path}')

def train_fish_detector(train_dir, val_dir, output_dir, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FishDetector(num_species=10)
    model.to(device)
    
    train_loader, val_loader = create_data_loaders(train_dir, val_dir)
    
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            bbox_output, class_output = model(images)
            
            class_loss = classification_criterion(class_output, labels)
            bbox_loss = regression_criterion(bbox_output, torch.randn_like(bbox_output))
            
            total_loss = class_loss + 0.1 * bbox_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                bbox_output, class_output = model(images)
                
                class_loss = classification_criterion(class_output, labels)
                bbox_loss = regression_criterion(bbox_output, torch.randn_like(bbox_output))
                total_loss = class_loss + 0.1 * bbox_loss
                
                val_loss += total_loss.item()
                
                _, predicted = torch.max(class_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(output_dir) / 'best_fish_detector.pth'
            torch.save(model.state_dict(), model_path)
    
    final_model_path = Path(output_dir) / 'final_fish_detector.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Training completed. Final model saved: {final_model_path}')

def main():
    parser = argparse.ArgumentParser(description='Train OceanMind models')
    parser.add_argument('--model', type=str, choices=['coral', 'fish', 'fishing'], required=True)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='trained_models')
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training {args.model} model...")
    print(f"Training directory: {args.train_dir}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.epochs}")
    
    if args.model == 'coral':
        train_coral_reef_model(args.train_dir, args.val_dir, args.output_dir, args.epochs)
    elif args.model == 'fish':
        train_fish_detector(args.train_dir, args.val_dir, args.output_dir, args.epochs)
    elif args.model == 'fishing':
        print("Fishing activity detection model training not implemented in this version.")

if __name__ == "__main__":
    main()