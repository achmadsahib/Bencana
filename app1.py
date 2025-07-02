import os
import cv2
import re
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
from flask import jsonify

# ==== Custom Layer ====
class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return [x * self.scale for x in inputs]
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({"scale": self.scale})
        return config

# ==== Flask Config ====
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ==== Load Models ====
custom_objects = {"CustomScaleLayer": CustomScaleLayer}
with custom_object_scope(custom_objects):
    cnn_model = load_model("models/cnn_custom_model.h5", compile=False)
    inception_model = load_model("models/inceptionv3_model.h5", compile=False)
    inception_resnet_model = load_model("models/inceptionresnet_model.h5", compile=False)

# Cek input shape CNN
cnn_input_shape = cnn_model.input_shape[1:3]  # e.g., (224, 224)

# ==== Kelas ====
CLASS_NAMES = ["Cyclone", "Earthquake", "Flood", "WildFire"]

# ==== Preprocessing Gambar ====
def preprocess_image(img, target_size=(299, 299)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def preprocess_for_cnn(img, target_size):
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==== Ensemble Predict (3 Model) ====
def ensemble_predict(img_pil):
    img_array_inception = preprocess_image(img_pil, target_size=(299, 299))
    img_array_cnn = preprocess_for_cnn(img_pil, target_size=cnn_input_shape)

    pred_cnn = cnn_model.predict(img_array_cnn)
    pred_inception = inception_model.predict(img_array_inception)
    pred_resnet = inception_resnet_model.predict(img_array_inception)

    print("CNN:", pred_cnn)
    print("InceptionV3:", pred_inception)
    print("InceptionResNetV2:", pred_resnet)

    ensemble = (pred_cnn + pred_inception + pred_resnet) / 3
    label = CLASS_NAMES[np.argmax(ensemble)]
    confidence = np.max(ensemble)
    return label, confidence

# ==== Proses Video ====
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)
        label, confidence = ensemble_predict(img_pil)
        predictions.append((label, confidence))

    cap.release()
    return predictions

# ==== Generate Filename Unik ====
def generate_filename(original_filename):
    ext = original_filename.rsplit(".", 1)[-1].lower()
    prefix = "gambar-upload" if ext in ["jpg", "jpeg", "png"] else "video-upload"

    existing_files = [
        f for f in os.listdir(app.config["UPLOAD_FOLDER"])
        if re.match(f"{prefix}-\\d+\\.{ext}", f)
    ]

    numbers = []
    for f in existing_files:
        match = re.search(rf"{prefix}-(\d+)\.{ext}", f)
        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers, default=0) + 1
    return f"{prefix}-{next_number}.{ext}"

# ==== Routing Halaman Utama ====
@app.route("/")
def index():
    return render_template("index.html")

# ==== Routing Prediksi ====
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({'error': 'Empty filename'}), 400

    filename = generate_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        if filename.lower().endswith(("jpg", "jpeg", "png")):
            img_pil = Image.open(filepath).convert("RGB")
            label, confidence = ensemble_predict(img_pil)
            return jsonify({
                'prediction': f"{label}"
            })

        elif filename.lower().endswith(("mp4", "avi", "mov")):
            predictions = process_video(filepath)
            pred_text = [f"Frame {i+1}: {label}"
                for i, (label, conf) in enumerate(predictions[:5])]

            return jsonify({
                'prediction': "<br>".join(pred_text)
            })

        return jsonify({'error': 'Unsupported file type'}), 400

    except Exception as e:
        print("Error:", e)  # Tampilkan di console
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

# ==== Jalankan Aplikasi ====
if __name__ == "__main__":
    app.run(debug=True, port=5001)

