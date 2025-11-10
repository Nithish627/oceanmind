import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path

class MarineDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = Path(image_dir)
        self.labels = labels
        self.transform = transform
        self.image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        filename = image_path.stem
        label = self.labels.get(filename, 0)
        
        return image, torch.tensor(label, dtype=torch.long)

class UnderwaterVideoLoader:
    def __init__(self, video_path, frame_skip=5):
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.cap = cv2.VideoCapture(video_path)
    
    def __iter__(self):
        self.frame_count = 0
        return self
    
    def __next__(self):
        for _ in range(self.frame_skip):
            ret = self.cap.grab()
            if not ret:
                raise StopIteration
        
        ret, frame = self.cap.retrieve()
        if not ret:
            raise StopIteration
        
        self.frame_count += 1
        return frame, self.frame_count
    
    def release(self):
        self.cap.release()

def create_data_loaders(train_dir, val_dir, batch_size=16):
    train_labels = {}
    val_labels = {}
    
    train_dataset = MarineDataset(train_dir, train_labels)
    val_dataset = MarineDataset(val_dir, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader