import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path

class IllegalFishingDetector(nn.Module):
    def __init__(self, num_classes=3):
        super(IllegalFishingDetector, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.temporal_net = nn.LSTM(in_features, 512, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        
        temporal_features, _ = self.temporal_net(features)
        output = self.classifier(temporal_features[:, -1, :])
        
        return output

class FishingActivityMonitor:
    def __init__(self, model_path=None, sequence_length=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IllegalFishingDetector()
        self.sequence_length = sequence_length
        self.frame_buffer = []
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.activity_labels = ["legal_fishing", "illegal_fishing", "no_activity"]
    
    def analyze_fishing_activity(self, frame_sequence):
        if len(frame_sequence) < self.sequence_length:
            return {"error": "Insufficient frames for analysis"}
        
        with torch.no_grad():
            processed_frames = [self.preprocess_frame(frame) for frame in frame_sequence[-self.sequence_length:]]
            frame_tensor = torch.stack(processed_frames).unsqueeze(0)
            
            output = self.model(frame_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            return {
                "activity": self.activity_labels[predicted_class],
                "confidence": probabilities[0][predicted_class].item(),
                "is_illegal": predicted_class == 1,
                "all_probabilities": probabilities.cpu().numpy()[0]
            }
    
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        return frame
    
    def add_frame(self, frame):
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.sequence_length * 2:
            self.frame_buffer.pop(0)
        
        if len(self.frame_buffer) >= self.sequence_length:
            return self.analyze_fishing_activity(self.frame_buffer)
        return None