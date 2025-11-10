import torch
import torch.nn as nn
import torchvision.models as models

class CoralReefDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(CoralReefDetector, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class CoralHealthAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CoralReefDetector()
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_image(self, image):
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            return {
                "health_status": predicted_class,
                "confidence": probabilities[0][predicted_class].item(),
                "probabilities": probabilities.cpu().numpy()[0]
            }
    
    def preprocess_image(self, image):
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)