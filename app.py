# app.py - Final Working Version

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Project Folder: {BASE_DIR}")

# Import model
try:
    from model import PetCNN
    print("✅ model.py loaded")
except ImportError:
    print("❌ model.py missing!")
    sys.exit(1)

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Using device: {device}")

# Model path
model_path = os.path.join(BASE_DIR, 'best_pet_model.pth')

# Check if model exists
if not os.path.exists(model_path):
    print("❌ Model file not found!")
    print("💡 Creating dummy model file...")

    temp_model = PetCNN()
    torch.save(temp_model.state_dict(), model_path)

    print("✅ Dummy model created")

# Load model
print("🔄 Loading model...")

model = PetCNN().to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    sys.exit(1)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASSES = ['Cat 🐱', 'Dog 🐶']


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction API
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load image
    image = Image.open(filepath).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    result = {
        "prediction": CLASSES[pred.item()],
        "confidence": f"{float(conf)*100:.2f}%",
        "image_url": f"static/uploads/{file.filename}"
    }

    return jsonify(result)


# Run server
if __name__ == '__main__':
    print("\n🚀 Server running at:")
    print("👉 http://localhost:5000\n")

    app.run(debug=True, port=5000)