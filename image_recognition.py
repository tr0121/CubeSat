import os
import torch
import torch.nn as nn
import time
import subprocess
from datetime import datetime
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from picamera import PiCamera

# Initialize camera
camera = PiCamera()

# GitHub repository details
REPO_DIR = "/home/pi/<YOUR_REPO>"  # Replace with your actual repo path
IMAGE_DIR = os.path.join(REPO_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)  # Ensure directory exists

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('wildfire_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Wildfire detection threshold
SIMILARITY_THRESHOLD = 0.7  # Adjust based on model performance

def capture_image():
    """Captures an image and returns the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.jpg")
    camera.capture(image_path)
    print(f"Captured {image_path}")
    return image_path

def predict_wildfire(image_path):
    """Runs the wildfire detection model on the captured image."""
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
    """Uploads an image to GitHub using a deploy key."""
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", image_path], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-m", f"Wildfire detected {datetime.now()}"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "push"], check=True)
        print("Wildfire image uploaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"GitHub upload error: {e}")

# Main loop
while True:
    image_path = capture_image()
    confidence = predict_wildfire(image_path)

    if confidence is not None:
        if confidence > SIMILARITY_THRESHOLD:
            print(f"Wildfire detected! Confidence: {confidence:.2f}")
            upload_to_github(image_path)
        else:
            print(f"No wildfire detected. Confidence: {confidence:.2f}. Deleting image.")
            os.remove(image_path)

    print("Waiting 6 minutes before next check...")
    time.sleep(360)  # Wait for 6 minutes before next iteration
