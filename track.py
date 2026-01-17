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

def midpoint(bbox: list) -> np.array:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

class Slice:
    bboxes = []
    positions = []

    def __init__(self, start_frame: int, bbox: np.ndarray):
        self.start_frame = start_frame
        self.last_frame = start_frame
        self.bboxes.append(bbox)
        self.positions.append(midpoint(bbox))

    def push(self, frame_index: int, bbox: np.ndarray):
        self.last_frame = frame_index
        self.bboxes.append(bbox)
        self.positions.append(midpoint(bbox))

    def append_slice(self, next_slice):
        if next_slice.last_frame <= self.last_frame:
            print("Warning: appending slice which occured before current.")

        for bbox, pos in zip(next_slice.bboxes, next_slice.positions):
            self.bboxes.append(bbox)
            self.positions.append(pos)

class RobotSlice(Slice):
    teamnum = '0'
    needs_ocr = True

    def contains(self, bbox: np.ndarray):
        x, y = midpoint(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        if y1 < y < y2 and x1 < x < x2:
            return True
        return False
    
    def setteam(self, team: str):
        self.teamnum = team
        self.needs_ocr = False

    def append_confidence(self, next_slice):
        confidence = 0.0 # Start at 0, conditions add more confidence
        if (next_slice.start_frame - 60) < self.last_frame: # Starts within 1 second
            confidence += 0.05    

tracking_model = get_roboflow_model(model_id="autoscout-qd2vx/7", api_key=os.getenv('ROBOFLOW_API_KEY'))

teamnums = ['2910', '1323', '4272', '2073', '4414', '1690']

reader = easyocr.Reader(['en'])

smoother = sv.DetectionsSmoother()
tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

slices = {}
balls = {}

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    global robot_detections
    results = tracking_model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    labels = ["" for _ in detections]

    for i, (class_id, tracker_id, bbox) in enumerate(zip(detections.class_id, detections.tracker_id, detections.xyxy)):
        match class_id:
            case 0:
                labels[i] = "Ball"
                if tracker_id not in balls:
                    balls[tracker_id] = Slice(frame_index, bbox)
                else:
                    balls[tracker_id].push(frame_index, bbox)
            case 1:
                if tracker_id not in slices:
                    slices[tracker_id] = RobotSlice(frame_index, bbox)
                else:
                    slices[tracker_id].push(frame_index, bbox)
                labels[i] = f"Team {slices[tracker_id].teamnum}"
            case 2:
                x1, y1, x2, y2 = map(int, bbox)
                cropped = frame[y1:y2, x1:x2]
                ocr_results = reader.readtext(cropped)
                if ocr_results:
                    if ocr_results[0][1] in teamnums:
                        labels[i] = f"{ocr_results[0][1]}"
                        for _, slice in slices.items():
                            if slice.needs_ocr and slice.contains(bbox):
                                slice.setteam(ocr_results[0][1])

            case _:
                print("Unknown classification")

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return annotated_frame

sv.process_video(
    source_path=sys.argv[1],
    target_path='labeled-'+sys.argv[1],
    callback=callback,
    show_progress=True
)
