import os
import torch
import torch.nn as nn
import time
import subprocess
from datetime import datetime
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from picamera2 import Picamera2

# --------------------- Configuration --------------------- #
REPO_DIR = os.path.expanduser("~/cubesat/CubeSat")
IMAGE_DIR = os.path.join(REPO_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

SIMILARITY_THRESHOLD = 0.7
MODE_FILE = os.path.join(REPO_DIR, "CubeSatmode.txt")

# --------------------- Initialize Camera --------------------- #
def setup_camera():
    """Initialize and configure the camera with error handling."""
    try:
        camera = Picamera2()
        config = camera.create_still_configuration()
        camera.configure(config)
        camera.start()
        print("✅ Camera initialized successfully")
        return camera
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        print("Check: 1) Camera cable connection 2) raspi-config camera enable 3) No other processes using camera")
        exit(1)

camera = setup_camera()

# --------------------- Load the Model --------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(
    os.path.join(REPO_DIR, "wildfire_model60.pth"),
    map_location=device
))
model = model.to(device)
model.eval()

# --------------------- Image Preprocessing --------------------- #
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------- Define Functions --------------------- #
def update_mode_file():
    """Pulls the latest mode file from GitHub."""
    try:
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
        print("Updated local mode file from GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating mode file: {e}")

def get_mode():
    """Reads mode file (active/passive)."""
    update_mode_file()
    try:
        with open(MODE_FILE, "r") as f:
            mode = f.read().strip().lower()
            return mode if mode in ["active", "passive"] else "active"
    except FileNotFoundError:
        return "active"

def capture_image():
    """Captures an image and returns its file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(IMAGE_DIR, f"image_{timestamp}.jpg")
    camera.capture_file(image_path)
    print(f"Captured {image_path}")
    return image_path

def predict_wildfire(image_path):
    """Processes the captured image with the wildfire detection model."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            confidence = torch.sigmoid(output).squeeze().item()
        return confidence
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Error processing image: {e}")
        return None

def upload_to_github(image_path):
    """Uploads an image to GitHub using your default SSH key."""
    try:
        # Set git safe directory
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", REPO_DIR], check=True)

        # SSH key configuration
        ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")
        if not os.path.exists(ssh_key_path):
            raise FileNotFoundError(f"SSH key missing: {ssh_key_path}")
        os.chmod(ssh_key_path, 0o600)

        # SSH agent setup
        subprocess.run(
            f"eval $(ssh-agent -s) && ssh-add {ssh_key_path}",
            shell=True,
            check=True,
            executable="/bin/bash"
        )

        # Convert to relative path
        rel_path = os.path.relpath(image_path, REPO_DIR)

        # Git operations
        subprocess.run(["git", "-C", REPO_DIR, "add", rel_path], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-m", f"Wildfire detected {datetime.now()}"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "push"], check=True)
        print("✅ Wildfire image uploaded successfully")

    except subprocess.CalledProcessError as e:
        print(f"❌ GitHub upload error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# --------------------- Main Loop --------------------- #
while True:
    try:
        mode = get_mode()
        print(f"🔄 Current mode: {mode}")

        if mode == "active":
            print("📸 Attempting image capture...")
            image_path = capture_image()

            print("🔮 Processing image for wildfire...")
            confidence = predict_wildfire(image_path)

            if confidence is not None:
                if confidence > SIMILARITY_THRESHOLD:
                    print(f"🔥 WILDFIRE DETECTED! Confidence: {confidence:.2f}")
                    upload_to_github(image_path)
                else:
                    print(f"✅ Clear. Confidence: {confidence:.2f}")
                    os.remove(image_path)
        else:
            print("🛑 Passive mode: Skipping capture")

        print("⏳ Next check in 10 seconds...")
        time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Script terminated by user")
        camera.stop()
        camera.close()
        exit(0)
