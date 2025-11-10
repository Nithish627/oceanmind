import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATHS = {
    "coral_reef": MODELS_DIR / "coral_reef_model.pth",
    "fish_detection": MODELS_DIR / "fish_detection_model.pth",
    "illegal_fishing": MODELS_DIR / "illegal_fishing_model.pth"
}

IMAGE_SIZE = (640, 640)
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

MARINE_SPECIES = [
    "clownfish", "blue_tang", "shark", "ray", "jellyfish",
    "sea_turtle", "dolphin", "whale", "octopus", "squid"
]

CORAL_HEALTH_LABELS = ["healthy", "bleached", "dead", "diseased"]

FISHING_ACTIVITY_LABELS = ["legal_fishing", "illegal_fishing", "no_activity"]