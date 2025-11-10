from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import io
import base64
from datetime import datetime
import json

app = Flask(__name__)

class OceanMindAPI:
    def __init__(self, coral_model, fish_model, fishing_model):
        self.coral_analyzer = coral_model
        self.fish_detector = fish_model
        self.fishing_monitor = fishing_model
        self.analysis_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_data = file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    coral_analysis = app.oceanmind.coral_analyzer.analyze_image(image)
    fish_detection = app.oceanmind.fish_detector.detect_marine_life(image)
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'coral_health': coral_analysis,
        'marine_life': fish_detection,
        'image_size': image.shape
    }
    
    app.oceanmind.analysis_history.append(result)
    
    return jsonify(result)

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    video_path = f"/tmp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    file.save(video_path)
    
    from processing.video_processor import VideoProcessor
    processor = VideoProcessor(
        app.oceanmind.coral_analyzer,
        app.oceanmind.fish_detector,
        app.oceanmind.fishing_monitor
    )
    
    results = processor.process_video(video_path)
    results['video_path'] = video_path
    
    return jsonify(results)

@app.route('/api/health/status')
def health_status():
    recent_analyses = app.oceanmind.analysis_history[-10:] if app.oceanmind.analysis_history else []
    
    health_summary = {
        'total_analyses': len(app.oceanmind.analysis_history),
        'recent_activities': recent_analyses,
        'system_status': 'operational',
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(health_summary)

@app.route('/api/species/count')
def species_count():
    if not app.oceanmind.analysis_history:
        return jsonify({'species_count': {}})
    
    species_count = {}
    for analysis in app.oceanmind.analysis_history[-100:]:
        species = analysis.get('marine_life', {}).get('species', 'unknown')
        species_count[species] = species_count.get(species, 0) + 1
    
    return jsonify({'species_count': species_count})

@app.route('/api/alerts/illegal_fishing')
def check_illegal_fishing():
    recent_activities = []
    
    for analysis in app.oceanmind.analysis_history[-50:]:
        if 'fishing_activity' in analysis:
            activity = analysis['fishing_activity']
            if activity.get('is_illegal', False):
                recent_activities.append({
                    'timestamp': analysis['timestamp'],
                    'confidence': activity['confidence'],
                    'location': 'detected_in_frame'
                })
    
    return jsonify({
        'illegal_activities_detected': len(recent_activities),
        'recent_alerts': recent_activities
    })

def initialize_api(coral_model, fish_model, fishing_model):
    app.oceanmind = OceanMindAPI(coral_model, fish_model, fishing_model)
    return app

if __name__ == '__main__':
    from models.coral_reef_detector import CoralHealthAnalyzer
    from models.fish_detector import MarineLifeTracker
    from models.illegal_fishing_detector import FishingActivityMonitor
    
    coral_analyzer = CoralHealthAnalyzer()
    fish_detector = MarineLifeTracker()
    fishing_monitor = FishingActivityMonitor()
    
    app = initialize_api(coral_analyzer, fish_detector, fishing_monitor)
    app.run(host='0.0.0.0', port=5000, debug=True)