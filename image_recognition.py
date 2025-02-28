import os
import torch
import torch.nn as nn
import time
import subprocess
from datetime import datetime
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from picamera2 import PiCamera2

# --------------------- Configuration --------------------- #
# GitHub repository details
REPO_DIR = "/tr0121/CubeSat"  # Replace with your actual repo path
IMAGE_DIR = os.path.join(REPO_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Wildfire detection threshold
SIMILARITY_THRESHOLD = 0.7  # Adjust based on model performance

# Mode file location (to control active/passive mode)
MODE_FILE = "/tr0121/CubeSatmode.txt" 

# --------------------- Initialize Camera --------------------- #
camera = PiCamera2()

# --------------------- Load the Model --------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('wildfire_model.pth', map_location=device))
model = model.to(device)
model.eval()

# --------------------- Image Preprocessing --------------------- #
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------- Define Functions --------------------- #
def update_mode_file():
    try:
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
        print("Updated local mode file from GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating mode file: {e}")

def get_mode():
    # Pull latest changes from GitHub so that mode.txt is updated
    update_mode_file()
    try:
        with open(MODE_FILE, "r") as f:
            mode = f.read().strip().lower()
            if mode in ["active", "passive"]:
                return mode
            else:
                return "active"
    except FileNotFoundError:
        return "active"


def capture_image():
    """ Captures an image and returns its file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.jpg")
    camera.capture(image_path)
    print(f"Captured {image_path}")
    return image_path

def predict_wildfire(image_path):
    """ Processes the captured image with the wildfire detection model."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            confidence = torch.sigmoid(output).item()  # Confidence score (0 to 1)
        return confidence
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Error processing image: {e}")
        return None

def upload_to_github(image_path):
    """ Uploads an image to GitHub using a deploy key."""
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", image_path], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-m", f"Wildfire detected {datetime.now()}"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "push"], check=True)
        print("Wildfire image uploaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"GitHub upload error: {e}")

# --------------------- Main Loop --------------------- #
while True:
    mode = get_mode()
    print(f"Current mode: {mode}")

    if mode == "active":
        image_path = capture_image()
        confidence = predict_wildfire(image_path)

        if confidence is not None:
            if confidence > SIMILARITY_THRESHOLD:
                print(f"Wildfire detected! Confidence: {confidence:.2f}")
                upload_to_github(image_path)
            else:
                print(f"No wildfire detected. Confidence: {confidence:.2f}. Deleting image.")
                os.remove(image_path)
    else:
        print("Passive mode active: Skipping image capture and processing.")

    print("Waiting 6 minutes before next check...")
    time.sleep(360)  # Wait for 6 minutes before next iteration
