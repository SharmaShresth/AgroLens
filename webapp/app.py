"""AgroLens backend.  Run:  pip install flask flask-cors tensorflow pillow numpy
                          python app.py   ->  open http://localhost:5000
Put your trained models in a 'saved_models' folder next to this file."""
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# Edit to match your notebooks. Class order/names live in index.html (PLANTS);
# this file only needs the model path, input size and preprocessing.
MODELS = {
    # All notebooks scale pixels with Rescaling(1/255) OUTSIDE the model -> preprocess "unit".
    "Tomato": {"path": "saved_models/tomato.keras", "size": 96, "preprocess": "unit"},
    "Rice":   {"path": "saved_models/rice.keras",   "size": 96, "preprocess": "unit"},
    "Radish": {"path": "saved_models/radish.keras", "size": 96, "preprocess": "unit"},
    "Potato": {"path": "saved_models/potato.keras", "size": 96, "preprocess": "unit"},
}
_loaded = {}


def get_model(plant):
    if plant not in _loaded:
        _loaded[plant] = keras.models.load_model(MODELS[plant]["path"])
    return _loaded[plant]


def prepare(img_bytes, cfg):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((cfg["size"], cfg["size"]))
    x = np.asarray(img, dtype="float32")
    if cfg["preprocess"] == "mobilenet":
        x = x / 127.5 - 1.0
    elif cfg["preprocess"] == "unit":
        x = x / 255.0
    return np.expand_dims(x, 0)


def get_classes(plant):
    """Read saved_models/<plant>_classes.txt (copy of class_names.txt from the notebook)."""
    path = MODELS[plant]["path"].replace(".keras", "_classes.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            return [l.strip() for l in fh if l.strip()]
    except OSError:
        return None


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    plant = request.form.get("plant")
    f = request.files.get("image")
    if plant not in MODELS or f is None:
        return jsonify(error="Bad request"), 400
    cfg = MODELS[plant]
    try:
        probs = get_model(plant).predict(prepare(f.read(), cfg), verbose=0)[0]
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(probs=[float(p) for p in probs], classes=get_classes(plant))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
