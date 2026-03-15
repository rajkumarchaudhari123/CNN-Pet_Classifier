# app.py

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from model import PetCNN

app = Flask(**name**)

# ----------------------------

# Paths

# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(**file**))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------

# Device

# ----------------------------

device = torch.device("cpu")
print("Using device:", device)

# ----------------------------

# Model path

# ----------------------------

MODEL_PATH = os.path.join(BASE_DIR, "best_pet_model.pth")

model = None

def load_model():
global model

```
if model is None:
    print("Loading model...")

    model = PetCNN()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )
    else:
        print("WARNING: model file not found")

    model.to(device)
    model.eval()

    print("Model ready")
```

# ----------------------------

# Image transforms

# ----------------------------

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])

CLASSES = ["Cat 🐱", "Dog 🐶"]

# ----------------------------

# Home route

# ----------------------------

@app.route("/")
def home():
return render_template("index.html")

# ----------------------------

# Prediction API

# ----------------------------

@app.route("/predict", methods=["POST"])
def predict():

```
try:

    load_model()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = Image.open(filepath).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

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

except Exception as e:

    return jsonify({
        "error": str(e)
    })
```

# ----------------------------

# Run server

# ----------------------------

if **name** == "**main**":

```
print("Server running at:")
print("http://localhost:5000")

app.run(host="0.0.0.0", port=5000)
```
