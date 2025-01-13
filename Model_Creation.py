import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import os
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Define the data transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Iterate through the folder structure and gather images
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.tif', '.png', '.jpg', '.jpeg')):  # Check for valid image files
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    
                    # Use folder name to assign labels
                    folder_name = os.path.basename(root).lower()
                    if folder_name == 'fire':  # Label for fire
                        self.labels.append(1)
                    elif folder_name == 'nofire':  # Label for no fire
                        self.labels.append(0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")  # Convert to RGB if necessary
        except (IOError, UnidentifiedImageError) as e:
            print(f"Skipping problematic image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # Skip the problematic image
        
        label = self.labels[idx]  # Get the corresponding label
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# 3. Load datasets and create data loaders
train_dir = r'C:\Users\benny\CubeSat\FlameVision\Classification\train'
val_dir = r'C:\Users\benny\CubeSat\FlameVision\Classification\valid'

train_dataset = WildfireDataset(train_dir, transform=transform)
val_dataset = WildfireDataset(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Define the model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Define the loss function and optimizer
num_fire = sum(label == 1 for label in train_dataset.labels)
num_nofire = sum(label == 0 for label in train_dataset.labels)

# Calculate the positive weight for handling class imbalance
pos_weight = torch.tensor([num_nofire / max(num_fire, 1)]).to(device)  # Avoid division by zero
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 6. Training loop
num_epochs = 2  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images = images.to(device)
        labels = labels.to(device)  # Move labels to device
        
        optimizer.zero_grad()  # Zero the gradients before each update
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())  # Squeeze to match label size
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights
        
        running_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.round(torch.sigmoid(outputs))  # Get predictions (0 or 1)
        correct_preds += (preds == labels).sum().item()  # Compare with actual labels
        total_preds += labels.size(0)
    
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_preds / total_preds
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

# Validation step
model.eval()  # Set model to evaluation mode
val_loss = 0.0
all_preds = []  # Initialize before validation
all_labels = []  # Initialize before validation

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())
        val_loss += loss.item()
        
        preds = torch.round(torch.sigmoid(outputs))  # Get predictions (0 or 1)
        all_preds.extend(preds.cpu().numpy().tolist())  # Collect all predictions
        all_labels.extend(labels.cpu().numpy().tolist())  # Collect all labels

# Compute metrics for the entire validation set
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

avg_val_loss = val_loss / len(val_loader)
val_accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# 7. Save the trained model
torch.save(model.state_dict(), 'wildfire_model.pth')
print("Model saved to 'wildfire_model.pth'")
print(f"Number of Wildfire samples: {sum(label == 1 for _, label in train_dataset)}")
print(f"Number of No Wildfire samples: {sum(label == 0 for _, label in train_dataset)}")
