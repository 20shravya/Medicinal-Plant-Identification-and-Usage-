# app.py
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json

BASE = Path(r"C:\MedicinalPlantApp")
MODELS_DIR = BASE / "models"
MODEL_PATH = MODELS_DIR / "plant_model_final.h5"   # adjust if your filename differs
CLASS_IDX_PATH = MODELS_DIR / "class_indices.json"
LEAF_INFO_PATH = MODELS_DIR / "leaf_info.json"

IMG_SIZE = 160

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load model once
print("Loading model...", MODEL_PATH)
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model loaded.")

# Load class mapping
if CLASS_IDX_PATH.exists():
    with open(CLASS_IDX_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    # invert mapping
    idx_to_class = {v: k for k, v in class_indices.items()}
else:
    idx_to_class = {}
    print("Warning: class_indices.json not found at", CLASS_IDX_PATH)

# Load leaf usage info
if LEAF_INFO_PATH.exists():
    with open(LEAF_INFO_PATH, "r", encoding="utf-8") as f:
        leaf_info = json.load(f)
else:
    leaf_info = {}
    print("Warning: leaf_info.json not found at", LEAF_INFO_PATH)

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

# -------------------------
# Routes: landing + main
# -------------------------

@app.route("/")
def intro():
    """
    Landing / Intro page (white background with the project title).
    Clicking the title should navigate to /home which loads the prediction page.
    """
    return render_template("intro.html")


@app.route("/home")
def home():
    """
    Main prediction page. This was your original index.html.
    """
    return render_template("index.html")


# -------------------------
# Prediction endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400
    f = request.files["image"]
    img_bytes = f.read()
    x = preprocess_image(img_bytes)
    preds = model.predict(x)[0]
    top_idx = int(np.argmax(preds))
    class_name = idx_to_class.get(top_idx, "Unknown")
    confidence = float(preds[top_idx])

    # get usage (list of paragraphs)
    entry = leaf_info.get(class_name, {})
    usage_list = entry.get("usage") if isinstance(entry.get("usage"), list) else []
    if not usage_list:
        # fallback: try usage_short / usage_long for backward compatibility
        if isinstance(entry.get("usage_short"), str):
            usage_list.append(entry.get("usage_short"))
        if isinstance(entry.get("usage_long"), str):
            usage_list.append(entry.get("usage_long"))
    if not usage_list:
        usage_list = ["No usage information available for this plant."]

    return jsonify({
        "predicted_class": class_name,
        "confidence": round(confidence, 4),
        "usage": usage_list
    })

if __name__ == "__main__":
    app.run(debug=True)
