import os
from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO('models/yolov8n.pt')

def calculate_percentage(count, max_count):
    return (count / max_count * 100) if max_count > 0 else 0

@app.route("/", methods=["GET", "POST"])
def home():
    signal_times = [30, 30, 30, 30]
    vehicles_counts = [0, 0, 0, 0]
    signal_status = ["Red", "Red", "Red", "Red"]
    ambulance_detected = [False, False, False, False]

    if request.method == "POST":
        uploaded_files = []
        
        for i in range(4):
            file = request.files.get(f"signal{i+1}")
            if file and file.filename != "":
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"signal{i+1}.jpg")
                file.save(image_path)
                uploaded_files.append(image_path)

                img = cv2.imread(image_path)
                if img is not None:
                    results = model(img)
                    result = results[0]
                    vehicles_counts[i] = len(result.boxes.xywh)
                    
                    # Check for ambulance using class ID (assuming 10 = ambulance in COCO dataset)
                    if any(cls == 10 for cls in result.boxes.cls):
                        ambulance_detected[i] = True

        # Handle ambulance priority
        if any(ambulance_detected):
            max_index = ambulance_detected.index(True)
            signal_status = ["Red"] * 4
            signal_status[max_index] = "Green"
            signal_times[max_index] = 60
        else:
            # Normal traffic management
            max_vehicles = max(vehicles_counts)
            max_index = vehicles_counts.index(max_vehicles)

            for i in range(4):
                if i == max_index:
                    signal_times[i] = 60
                    signal_status[i] = "Green"
                else:
                    signal_times[i] = 30
                    signal_status[i] = "Red"

            next_signal = (max_index + 1) % 4
            signal_status[next_signal] = "Yellow"

    max_vehicle_count = max(vehicles_counts) if any(vehicles_counts) else 1
    percentages = [calculate_percentage(count, max_vehicle_count) for count in vehicles_counts]

    return render_template(
        "index.html",
        signal_times=signal_times,
        vehicles_counts=vehicles_counts,
        signal_status=signal_status,
        percentages=percentages,
        upload_folder=UPLOAD_FOLDER
    )

if __name__ == "__main__":
    app.run(debug=True)
