from flask import Flask, request, jsonify
import joblib
import torch
from DGMHTFTLoader import DGMHTFT
from pytorch_forecasting.metrics import QuantileLoss
# === Flask app ===
app = Flask(__name__)

MODEL_PATH = "./Model/best_model_epoch=05_val_loss=183.57.ckpt"
DATASET_PATH = "./Model/training_dataset_obj.pkl"

# Load dataset object
try:
    training_dataset_obj = joblib.load(DATASET_PATH)
    print("✅ Training dataset object berhasil dimuat!")
except Exception as e:
    training_dataset_obj = None
    print(f"❌ Error memuat training dataset object: {e}")

# === Load model langsung dari checkpoint di CPU tanpa memanggil from_dataset ===
try:
    best_model = DGMHTFT.load_from_checkpoint(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        strict=False
    )

    # Optional: pindahkan semua logging_metrics ke CPU
    if hasattr(best_model, "logging_metrics"):
        for metric in best_model.logging_metrics:
            metric.to("cpu")

    best_model.to("cpu")
    best_model.eval()
    print("✅ Model berhasil dimuat di CPU")
except Exception as e:
    best_model = None
    print(f"❌ Gagal load model: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    if training_dataset_obj is None or best_model is None:
        return jsonify({"error": "Training dataset object atau model tidak tersedia"}), 500

    try:
        data_json = request.get_json()
        if not data_json:
            return jsonify({"error": "Data JSON tidak ditemukan"}), 400

        # === TODO: logika prediksi di sini ===
        return jsonify({"message": "Prediksi berhasil", "data": data_json})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
