import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Load class names
from torchvision.datasets import ImageFolder
class_names = ImageFolder('dataset').classes

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('models/lazy_landmark_model_best.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Lazy Landmark Finder")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        pred_class = class_names[predicted.item()]
    st.write(f"**Predicted Landmark:** {pred_class}")
