from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import os
import json
from ANN import TinyMLP

app = Flask(__name__)
CORS(app)  # 允许跨域请求

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with open(os.path.join(UPLOAD_FOLDER, "weights.json"), "r") as f:
    weights = json.load(f)
model = TinyMLP.from_dict(weights)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    pixels = [(255 - px) / 255.0 * 0.99 + 0.01 for px in img.getdata()]
    return pixels

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, "uploaded.png")
    file.save(img_path)

    try:
        x = preprocess_image(img_path)
        pred, _ = model.predict(x)
        return jsonify({"prediction": int(pred)})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # 允许外部访问
