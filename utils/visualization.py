import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_detections(image, detections, color_map=None):
    if color_map is None:
        color_map = {
            'fish': (0, 255, 0),
            'coral': (255, 0, 0),
            'illegal_fishing': (0, 0, 255)
        }
    
    result_image = image.copy()
    
    for detection in detections:
        bbox = detection.get('bbox', [])
        label = detection.get('label', 'unknown')
        confidence = detection.get('confidence', 0)
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            color = color_map.get(label, (255, 255, 255))
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_image

def create_health_visualization(health_data):
    labels = ['Healthy', 'Bleached', 'Dead', 'Diseased']
    probabilities = health_data.get('probabilities', [0.25, 0.25, 0.25, 0.25])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, probabilities, color=['green', 'yellow', 'red', 'orange'])
    
    ax.set_ylabel('Probability')
    ax.set_title('Coral Reef Health Assessment')
    ax.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_marine_life_distribution(species_data):
    species = list(species_data.keys())
    counts = list(species_data.values())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(species, counts, color='skyblue')
    
    ax.set_xlabel('Marine Species')
    ax.set_ylabel('Count')
    ax.set_title('Marine Life Distribution')
    plt.xticks(rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_activity_timeline(activity_data):
    timestamps = [item['timestamp'] for item in activity_data]
    activities = [item['activity'] for item in activity_data]
    confidences = [item['confidence'] for item in activity_data]
    
    activity_colors = {
        'legal_fishing': 'green',
        'illegal_fishing': 'red',
        'no_activity': 'blue'
    }
    
    colors = [activity_colors.get(activity, 'gray') for activity in activities]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    scatter = ax.scatter(timestamps, confidences, c=colors, s=100, alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Confidence')
    ax.set_title('Fishing Activity Timeline')
    ax.grid(True, alpha=0.3)
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=label, markersize=10)
                      for label, color in activity_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig