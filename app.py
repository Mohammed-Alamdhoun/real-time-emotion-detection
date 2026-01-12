from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import base64
import csv
import os
from datetime import datetime
from scipy.stats import entropy
import re
import tensorflow as tf

# --------- Flask setup --------- #
app = Flask(__name__, template_folder="templates")
CORS(app)

# ------------------- List all GPUs ------------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

    try:
        # ------------------- Set memory growth ------------------- #
        # This makes TF allocate memory gradually as needed instead of pre-allocating all
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # ------------------- Optional: Limit GPU memory ------------------- #
        # Example: Limit first GPU to 4GB
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) configured.")

    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU.")

# --------- Load the model and labels --------- #
model = load_model('models/best_model_finetune_v1.h5')
labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']

# --------- Load face detector (OpenCV DNN) --------- #
prototxt_path = "models/deploy.prototxt"
weights_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# --------- Global variables for session logging --------- #
current_session_id = None
current_csv_file = None

# Directory to save session CSVs
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# --------- Helper functions --------- #
def get_next_session_number():
    """
    Determine the next session number based on existing files in the sessions folder.
    """
    existing_files = os.listdir(SESSIONS_DIR)
    session_numbers = []

    for filename in existing_files:
        match = re.match(r"session_(\d+)\.csv", filename)
        if match:
            session_numbers.append(int(match.group(1)))

    return max(session_numbers) + 1 if session_numbers else 1


def save_prediction_to_csv(emotion, preds, max_prob, confidence_score, prob_entropy):
    """
    Append prediction results + features to the active CSV session file.
    """
    if current_csv_file is None:
        return

    with open(current_csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.now().isoformat(), emotion] +
            list(map(float, preds)) +
            [max_prob, confidence_score, prob_entropy]
        )

# --------- Routes --------- #
@app.route('/')
def index():
    """Serve the frontend page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive base64 image from frontend,
    detect face, predict emotion,
    calculate extra features, and save to CSV if recording.
    """
    try:
        # Parse image from request
        data = request.get_json(force=True)
        image_data = data.get("image", None)
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image to numpy array
        img_bytes = base64.b64decode(image_data.split(",")[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Couldn't decode image"}), 400

        (h, w) = frame.shape[:2]

        # Detect faces using OpenCV DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        # If no detections
        if detections.shape[2] == 0:
            return jsonify({"error": "No face detected"}), 400

        # Pick the face with the highest confidence > 0.7
        max_conf = 0
        best_face = None
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > max_conf and confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                best_face = (
                    max(0, x),
                    max(0, y),
                    min(w - 1, x2),
                    min(h - 1, y2)
                )

        if best_face is None:
            return jsonify({"error": "No face detected"}), 400

        (x, y, x2, y2) = best_face
        face = frame[y:y + (y2 - y), x:x + (x2 - x)]
        if face.size == 0:
            return jsonify({"error": "Empty face crop"}), 400

        # Preprocess face for model
        face_resized = cv2.resize(face, (112, 112))
        face_resized = img_to_array(face_resized).astype("float32")
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict emotion
        preds = model.predict(face_resized)[0]
        emotion_idx = int(np.argmax(preds))
        emotion = labels[emotion_idx]

        # --------- Calculate extra features --------- #
        max_prob = float(np.max(preds))
        sorted_probs = np.sort(preds)[::-1]
        second_max_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        confidence_score = max_prob - second_max_prob
        prob_entropy = float(entropy(preds))

        # --------- Save to CSV if session active --------- #
        save_prediction_to_csv(emotion, preds, max_prob, confidence_score, prob_entropy)

        # Send response to frontend
        return jsonify({
            "prediction": emotion,
            "box": [int(x), int(y), int(x2), int(y2)]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/start_session', methods=['POST'])
def start_session():
    """
    Start a new recording session.
    Automatically increment session number based on existing files.
    """
    global current_session_id, current_csv_file

    # Get the next session number
    session_number = get_next_session_number()
    current_session_id = f"session_{session_number}"

    # Create CSV file path
    filename = f"{current_session_id}.csv"
    current_csv_file = os.path.join(SESSIONS_DIR, filename)

    # Write CSV header
    with open(current_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "emotion"] +
            labels +
            ["max_prob", "confidence_score", "entropy"]
        )

    return jsonify({"message": f"Session {session_number} started", "session_id": current_session_id}), 200


@app.route('/stop_session', methods=['POST'])
def stop_session():
    """
    Stop the current recording session.
    """
    global current_session_id, current_csv_file

    if current_session_id is None:
        return jsonify({"error": "No active session"}), 400

    session_id = current_session_id
    current_session_id = None
    current_csv_file = None

    return jsonify({"message": f"Session {session_id} stopped"}), 200

# --------- Run Flask app --------- #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)