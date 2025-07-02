import torch 
from torchvision import models, transforms
from PIL import Image
import sys
import os

MODEL_PATH = '../models/lazy_landmark_best.pth'
DATASET_DIR = 'dataset'
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if IMG_PATH is None or not os.path.isfile(IMG_PATH):
    print('Usage: python src/infer.py path_to_image.jpg')
    sys.exit(1)

#load class names
from torchvision.datasets import ImageFolder
dummy_dataset = ImageFolder(DATASET_DIR)
classes_names = dummy_dataset.classes

#defining transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

#loading preprocessing the images
img = Image.open(IMG_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

#load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features,len(classes_names))
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model = model.to(device)
model.eval()

#inference
with torch.no_grad():
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    _,predicted =outputs.max(1)
    pred_class = classes_names[predicted.item()]

print(f'Predicted landmark: {pred_class}')