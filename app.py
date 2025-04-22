import logging
import os
import cv2
from flask import Flask, request, send_file, make_response
from ultralytics import YOLO
import tempfile
import datetime
from pathlib import Path

# Set up Flask app
app = Flask(__name__)

# Define directories and model path
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "models" / "best.pt"
model = YOLO(str(model_path))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define confidence threshold
confidence_threshold = 0.12

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a sequential log file name
log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Set up logging configuration
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(message)s")

# Function to check for overlap between two bounding boxes
def check_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right <= x_left or y_bottom <= y_top:
        return False
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area > 0.5 * min(w1 * h1, w2 * h2)

# Process the video to detect smokers
def process_video(input_path, output_path, frame_skip=2, filename="uploaded_video.mp4"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_annotated = None
    smoker_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame)
            annotated_frame = results[0].plot()
            gray = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                for result in results[0].boxes:
                    if result.cls == 0 and result.conf.item() >= confidence_threshold:
                        cx, cy, cw, ch = result.xywh[0]
                        cigarette_box = (cx - cw / 2, cy - ch / 2, cw, ch)
                        if check_overlap((x, y, w, h), cigarette_box):
                            smoker_found = True
                            cv2.rectangle(annotated_frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 255), 4)
                            cv2.putText(annotated_frame, "Smoker Detected", (x, y - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            # Log cigarette detection with timestamp
                            logging.info(f"Cigarette detected at frame {frame_idx}, face at ({x},{y}), "
                                         f"confidence: {result.conf.item():.2f}")
                            break
                if not smoker_found:
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            last_annotated = annotated_frame.copy()
        else:
            annotated_frame = last_annotated.copy() if last_annotated is not None else frame

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

# Flask route to upload video
@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return make_response("No video uploaded", 400)

    file = request.files['video']
    if file.filename == '':
        return make_response("No file selected", 400)

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        input_path = temp_input.name
        file.save(input_path)

    # Create temp file for output video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = temp_output.name

    # Process the video
    process_video(input_path, output_path, filename=file.filename)

    # Send back the processed video
    response = make_response(send_file(output_path, mimetype='video/mp4', as_attachment=True, download_name='processed_video.mp4'))

    os.unlink(input_path)  # Delete temp input file
    
    # Provide log file URL back as a response (for simplicity, return the filename)
    log_file_url = f"{log_file}"
    response.headers["Log-File"] = log_file_url
    
    return response

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)