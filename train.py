import os
import json
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "Images"
MODELS_DIR = ROOT_DIR / "models"
LABELS_PATH = ROOT_DIR / "labels.json"

# Define the PyTorch model
class BanglaOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(BanglaOCRModel, self).__init__()
        # Conv2d: 1 input channel (grayscale), 32 output channels, kernel size 3, same padding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Max-Pooling: kernel size 2, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv2d: 32 input channels, 64 output channels, kernel size 3, same padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # After two MaxPools on 32x32: 32 -> 16 -> 8
        # Flattened features: 64 channels * 8 * 8 spatial dimension
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # Fully connected layer to num_classes output logits
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} does not exist. Please place the dataset there.")
        return

    # Hyperparameters
    batch_size = 64
    img_height = 32
    img_width = 32
    epochs = 5
    lr = 0.0005

    # Define transforms (Grayscale, resize to 32x32, normalize to [0,1] tensor)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(), # handles the rescaling / 255.0 automatically
    ])

    # Load dataset
    print("Loading dataset...")
    full_dataset = ImageFolder(root=str(DATA_DIR), transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")

    # Save labels mapping
    labels_dict = {i: name for i, name in enumerate(class_names)}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_dict, f)
    print(f"Saved labels to {LABELS_PATH}")

    # Reproducible train/validation split (80/20) with fixed seed
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(123)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model, loss, and optimizer
    model = BanglaOCRModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bangla-ocr")
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("img_size", img_height)
        mlflow.log_param("device", str(device))

        print("Training model...")
        val_epoch_acc = 0.0
        val_epoch_loss = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print progress update every 500 batches to keep logs minimal but informative
                if (batch_idx + 1) % 500 == 0:
                    current_loss = running_loss / total
                    current_acc = correct / total
                    print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] "
                          f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}")
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
            
            # Log metrics per epoch
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_epoch_acc, step=epoch)

        # Log final validation accuracy and loss matching original keys
        mlflow.log_metric("val_accuracy", val_epoch_acc)
        mlflow.log_metric("val_loss", val_epoch_loss)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Model logged to MLflow.")

if __name__ == "__main__":
    main()
