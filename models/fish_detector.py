import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path

class FishDetector(nn.Module):
    def __init__(self, num_species=10):
        super(FishDetector, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        in_features = self.backbone.classifier[0].in_features
        
        self.backbone.classifier = nn.Identity()
        
        self.detection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_species)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        bbox_output = self.detection_head(features)
        class_output = self.classification_head(features)
        return bbox_output, class_output

class MarineLifeTracker:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FishDetector()
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.species_list = [
            "clownfish", "blue_tang", "shark", "ray", "jellyfish",
            "sea_turtle", "dolphin", "whale", "octopus", "squid"
        ]
    
    def detect_marine_life(self, image):
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            bbox_output, class_output = self.model(image_tensor)
            
            species_probs = torch.softmax(class_output, dim=1)
            predicted_species = torch.argmax(species_probs, dim=1).item()
            
            bbox = bbox_output[0].cpu().numpy()
            
            return {
                "species": self.species_list[predicted_species],
                "species_confidence": species_probs[0][predicted_species].item(),
                "bounding_box": bbox.tolist(),
                "all_species_probabilities": species_probs.cpu().numpy()[0]
            }
    
    def preprocess_image(self, image):
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)
    
    def track_multiple_frames(self, frames):
        detections = []
        for frame in frames:
            detection = self.detect_marine_life(frame)
            detections.append(detection)
        return detections