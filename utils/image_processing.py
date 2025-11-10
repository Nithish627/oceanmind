import cv2
import numpy as np
import torch

def enhance_underwater_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def remove_water_artifacts(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=~mask)
    
    return result

def detect_objects_yolo(image, model, confidence_threshold=0.5):
    results = model(image)
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf.item()
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls.item())
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id
                })
    
    return detections

def create_heatmap(detections, image_shape):
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection['confidence']
        
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        radius = int(min(x2 - x1, y2 - y1) / 4)
        cv2.circle(heatmap, (center_x, center_y), radius, confidence, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    return heatmap.astype(np.uint8)

def optical_flow_motion_detection(prev_frame, current_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    motion_mask = magnitude > 2.0
    
    return motion_mask, magnitude