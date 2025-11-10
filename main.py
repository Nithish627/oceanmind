import argparse
import cv2
import json
from datetime import datetime
from pathlib import Path

from models.coral_reef_detector import CoralHealthAnalyzer
from models.fish_detector import MarineLifeTracker
from models.illegal_fishing_detector import FishingActivityMonitor
from processing.video_processor import VideoProcessor, RealTimeProcessor
from api.app import initialize_api
import torch

def main():
    parser = argparse.ArgumentParser(description='OceanMind Marine Ecosystem Monitoring')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'realtime', 'api'], 
                       required=True, help='Operation mode')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    
    args = parser.parse_args()
    
    print("Initializing OceanMind AI System...")
    
    coral_analyzer = CoralHealthAnalyzer()
    fish_detector = MarineLifeTracker()
    fishing_monitor = FishingActivityMonitor()
    
    if args.mode == 'image':
        process_image(args.input, args.output, coral_analyzer, fish_detector)
    
    elif args.mode == 'video':
        process_video(args.input, args.output, coral_analyzer, fish_detector, fishing_monitor)
    
    elif args.mode == 'realtime':
        process_realtime(coral_analyzer, fish_detector, fishing_monitor)
    
    elif args.mode == 'api':
        start_api_server(coral_analyzer, fish_detector, fishing_monitor)

def process_image(input_path, output_dir, coral_analyzer, fish_detector):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image from {input_path}")
        return
    
    coral_analysis = coral_analyzer.analyze_image(image)
    fish_detection = fish_detector.detect_marine_life(image)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'input_file': input_path,
        'coral_health_analysis': coral_analysis,
        'marine_life_detection': fish_detection
    }
    
    with open(output_path / f'analysis_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis completed. Results saved to {output_path / f'analysis_{timestamp}.json'}")

def process_video(input_path, output_dir, coral_analyzer, fish_detector, fishing_monitor):
    processor = VideoProcessor(coral_analyzer, fish_detector, fishing_monitor)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = output_path / f'processed_{timestamp}.avi'
    output_json = output_path / f'results_{timestamp}.json'
    
    print(f"Processing video: {input_path}")
    results = processor.process_video(input_path, str(output_video))
    
    processor.save_results(str(output_json))
    
    print(f"Video processing completed.")
    print(f"Output video: {output_video}")
    print(f"Analysis results: {output_json}")

def process_realtime(coral_analyzer, fish_detector, fishing_monitor):
    processor = RealTimeProcessor(coral_analyzer, fish_detector, fishing_monitor)
    
    print("Starting real-time monitoring...")
    print("Press 'q' to quit.")
    
    processor.start_stream_processing()

def start_api_server(coral_analyzer, fish_detector, fishing_monitor):
    from api.app import initialize_api
    app = initialize_api(coral_analyzer, fish_detector, fishing_monitor)
    
    print("Starting OceanMind API server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()