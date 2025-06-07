import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------- Dataset ----------
class PNGDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_samples=100):
        self.data = pd.read_csv(csv_file).drop_duplicates("patientId")
        self.img_dir = img_dir
        self.transform = transform
        self.data = self.data[:max_samples]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pid = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, -1]
        
        img_path = os.path.join(self.img_dir, pid + ".png")
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, torch.tensor([label], dtype=torch.float32)

# ---------- Model ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*64*64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------- Main Training ----------
def main():
    # Paths
    CSV_PATH = "lesson3-data//stage_2_train_labels.csv"
    IMG_DIR = "png_images/"
    MODEL_PATH = "cnn_model.pth"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = PNGDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Model
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(10):  # Just 2 epochs for testing
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")
    
    print("Training done!")
    
    # ---------- Save the Model ----------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()