from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2

# Initialize Flask
app = Flask(__name__)

# Folder to save uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load your trained YOLO model
MODEL_PATH = 'model/best.pt'
model = YOLO(MODEL_PATH)

print("✅ YOLO Model Loaded Successfully!")
print("Classes:", model.names)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Run YOLO prediction
    results = model(img_path)

    img = cv2.imread(img_path)

    boxes = results[0].boxes

    if len(boxes) > 0:

        # Pick the box with highest confidence
        best_box = max(boxes, key=lambda b: float(b.conf[0]))

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        conf = float(best_box.conf[0])

        # ----------- GET CLASS NAME -----------
        cls_id = int(best_box.cls[0])
        label = model.names[cls_id]
        # --------------------------------------

        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        result_filename = f"analyzed_{file.filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)

        confidence = round(conf * 100, 2)

    else:
        # No fracture found → healthy
        result_filename = f"analyzed_{file.filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)

        label = "Healthy"
        confidence = 0

    # Return data to frontend
    return jsonify({
        "original_image": f"/static/uploads/{file.filename}",
        "analyzed_image": f"/static/results/{result_filename}",
        "label": label,
        "confidence": confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
