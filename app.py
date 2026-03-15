from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from model import PetCNN

app = Flask(__name__)

# ----------------------------
# Paths
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    
    if model is None:
        print("Loading model...")
        
        model = PetCNN()
        
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location=device)
            )
            print("✅ Model loaded successfully!")
        else:
            print("❌ WARNING: model file not found at:", MODEL_PATH)
            print("   Please train the model first using train.py")
        
        model.to(device)
        model.eval()
        
        print("Model ready")

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
    
    try:
        # Load model (if not already loaded)
        load_model()
        
        # Check if model was loaded properly
        if model is None:
            return jsonify({"error": "Model not loaded. Please train the model first using train.py"})
        
        # Check if file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"})
        
        # Check if it's an image
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Please upload an image file (jpg, jpeg, png, gif)"})
        
        # Save the file
        filename = file.filename
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )
        
        file.save(filepath)
        print(f"✅ File saved: {filename}")
        
        # Open and process image
        image = Image.open(filepath).convert("RGB")
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        # Get confidence percentage
        confidence = float(conf) * 100
        
        # Prepare result
        result = {
            "success": True,
            "prediction": CLASSES[pred.item()],
            "confidence": f"{confidence:.2f}%",
            "image_url": f"static/uploads/{filename}"
        }
        
        print(f"✅ Prediction: {result['prediction']} with {result['confidence']} confidence")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ----------------------------
# Run server
# ----------------------------

if __name__ == "__main__":
    
    # Load model at startup
    load_model()
    
    print("\n" + "="*50)
    print("🚀 Server starting...")
    print("📱 Local: http://localhost:5000")
    print("🌍 Network: http://0.0.0.0:5000")
    print("="*50 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)