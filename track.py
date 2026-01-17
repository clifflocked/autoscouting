import os
import sys
import dotenv

dotenv.load_dotenv()

if os.getenv('ROBOFLOW_API_KEY') == None:
    print("No API key found. Check your .env")
    exit(1)

if len(sys.argv) != 2:
    print("Usage: track.py VIDEO.mp4")
    exit(1)

# Do these later since they take more time
import easyocr
import numpy as np
import supervision as sv
from inference.models.utils import get_roboflow_model

tracking_model = get_roboflow_model(model_id="autoscout-qd2vx/7", api_key=os.getenv('ROBOFLOW_API_KEY'))

teamnums = ['2910', '1323', '4272', '2073', '4414', '1690']

reader = easyocr.Reader(['en'])

smoother = sv.DetectionsSmoother()
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

robot_detections = {}

def classid_to_str(classid: int):
    if classid == 0:
        return "ball"
    elif classid == 1:
        return "robot"
    elif classid == 2:
        return "teamnum"
    else:
        return "unknown"

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    global robot_detections
    results = tracking_model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    # Pre-populate labels for all detections
    labels = [classid_to_str(cid) for cid in detections.class_id]

    for i, (class_id, tracker_id, bbox) in enumerate(zip(detections.class_id, detections.tracker_id, detections.xyxy)):
        if class_id != 2:
            continue # Not a team number

        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)

        # Crop the frame to the team number region
        cropped = frame[y1:y2, x1:x2]

        # Skip if crop is invalid
        if cropped.size == 0:
            continue

        # Run OCR on the cropped team number
        ocr_results = reader.readtext(cropped)

        # Only label if OCR result directly matches a known team number
        if ocr_results:
            text = ocr_results[0][1]
            if text in teamnums:
                labels[i] = f"Team {text}"

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return annotated_frame

sv.process_video(
    source_path=sys.argv[1],
    target_path='labeled-'+sys.argv[1],
    callback=callback,
    show_progress=True
)
