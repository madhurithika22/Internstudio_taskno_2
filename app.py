import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

# ---------- Load Model ----------
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device("cpu")))
model.eval()

# ---------- Prediction ----------
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        prediction = "Pneumonia" if output.item() > 0.5 else "Normal"
        confidence = output.item()
    
    return prediction, confidence
st.title("MedAI - Detects Pneumonia with your x-ray")

uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    pred, conf = predict(image)
    st.write(f"**Prediction:** {pred}")
    st.write(f"**Confidence:** {conf:.4f}")