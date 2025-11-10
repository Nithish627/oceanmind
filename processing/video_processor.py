import cv2
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

class VideoProcessor:
    def __init__(self, coral_model, fish_model, fishing_model):
        self.coral_analyzer = coral_model
        self.fish_detector = fish_model
        self.fishing_monitor = fishing_model
        
        self.results = {
            'coral_health': [],
            'fish_detections': [],
            'fishing_activities': [],
            'timestamps': []
        }
    
    def process_video(self, video_path, output_path=None, frame_skip=10):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        processed_count = 0
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 640))
        else:
            out = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                processed_frame = self.process_frame(frame, frame_count)
                processed_count += 1
                
                if out and processed_frame is not None:
                    out.write(processed_frame)
            
            frame_count += 1
        
        cap.release()
        if out:
            out.release()
        
        return self.results
    
    def process_frame(self, frame, frame_number):
        timestamp = datetime.now().isoformat()
        
        frame_resized = cv2.resize(frame, (640, 640))
        
        coral_analysis = self.coral_analyzer.analyze_image(frame_resized)
        fish_detection = self.fish_detector.detect_marine_life(frame_resized)
        fishing_activity = self.fishing_monitor.add_frame(frame_resized)
        
        self.results['coral_health'].append(coral_analysis)
        self.results['fish_detections'].append(fish_detection)
        self.results['timestamps'].append(timestamp)
        
        if fishing_activity:
            self.results['fishing_activities'].append(fishing_activity)
        
        annotated_frame = self.annotate_frame(frame_resized, coral_analysis, fish_detection)
        
        return annotated_frame
    
    def annotate_frame(self, frame, coral_analysis, fish_detection):
        annotated = frame.copy()
        
        health_status = coral_analysis['health_status']
        health_labels = ['Healthy', 'Bleached', 'Dead', 'Diseased']
        health_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (0, 165, 255)]
        
        cv2.putText(annotated, f"Coral: {health_labels[health_status]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, health_colors[health_status], 2)
        cv2.putText(annotated, f"Confidence: {coral_analysis['confidence']:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        species = fish_detection['species']
        species_confidence = fish_detection['species_confidence']
        
        cv2.putText(annotated, f"Species: {species}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Det Confidence: {species_confidence:.2f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        bbox = fish_detection.get('bounding_box', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return annotated
    
    def save_results(self, output_path):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return output_path

class RealTimeProcessor:
    def __init__(self, coral_model, fish_model, fishing_model):
        self.coral_analyzer = coral_model
        self.fish_detector = fish_model
        self.fishing_monitor = fishing_model
        self.is_running = False
    
    def start_stream_processing(self, stream_url=0):
        self.is_running = True
        cap = cv2.VideoCapture(stream_url)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_realtime_frame(frame)
            cv2.imshow('OceanMind Real-time Monitoring', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_realtime_frame(self, frame):
        frame_resized = cv2.resize(frame, (640, 640))
        
        coral_analysis = self.coral_analyzer.analyze_image(frame_resized)
        fish_detection = self.fish_detector.detect_marine_life(frame_resized)
        
        annotated_frame = self.annotate_realtime_frame(frame_resized, coral_analysis, fish_detection)
        
        return annotated_frame
    
    def annotate_realtime_frame(self, frame, coral_analysis, fish_detection):
        annotated = frame.copy()
        
        health_labels = ['Healthy', 'Bleached', 'Dead', 'Diseased']
        health_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255), (0, 165, 255)]
        
        health_status = coral_analysis['health_status']
        cv2.rectangle(annotated, (0, 0), (300, 130), (0, 0, 0), -1)
        
        cv2.putText(annotated, f"Coral Health: {health_labels[health_status]}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, health_colors[health_status], 2)
        cv2.putText(annotated, f"Confidence: {coral_analysis['confidence']:.2f}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        species = fish_detection['species']
        cv2.putText(annotated, f"Detected: {species}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Species Conf: {fish_detection['species_confidence']:.2f}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        alert_text = ""
        if coral_analysis['health_status'] in [1, 2, 3]:
            alert_text = "ALERT: Coral Health Issue!"
            cv2.putText(annotated, alert_text, (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    
    def stop_processing(self):
        self.is_running = False