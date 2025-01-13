import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Load the model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)  # Define the model architecture
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjust the output layer for binary classification
model.load_state_dict(torch.load('wildfire_model.pth'))  # Load the weights into the model
model = model.to(device)  # Move the model to the appropriate device
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match training normalization
])

# Define the similarity threshold
similarity_threshold = 0.7  # Adjust based on experimentation

# Load and preprocess the image
img_path = "FlatSat_Student.py/image.png"
try:
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # Apply transformations and add batch dimension

    # Get model output
    with torch.no_grad():
        output = model(img)  # Logits
        confidence = torch.sigmoid(output).item()  # Confidence score (0 to 1)

    # Classify based on confidence
    if confidence > similarity_threshold:
        print(f"Prediction: Wildfire (Confidence: {confidence:.2f})")
    else:
        print(f"Prediction: No Wildfire (Confidence: {confidence:.2f})")

except FileNotFoundError:
    print(f"Error: File {img_path} not found.")
except UnidentifiedImageError:
    print(f"Error: File {img_path} is not a valid image.")







