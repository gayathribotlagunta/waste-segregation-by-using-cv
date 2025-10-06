import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Configuration / Globals
WEIGHTS = "yolov12n.pt"  # or path to your trained weights
IMAGE_SIZE = 640  # input size
CLASS_NAMES = ["biodegradable", "non_biodegradable", "recyclable"]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Initialize Model
def load_model():
    # load the YOLOv12 model
    model = YOLO(WEIGHTS)
    # configure thresholds
    model.conf = CONF_THRESHOLD
    model.iou = IOU_THRESHOLD
    return model

# Inference & Annotation
def infer_image(model, img_path, show=True, save_path=None):
    img = cv2.imread(img_path)
    if img is None:
        print("Error reading image:", img_path)
        return

    results = model(img, size=IMAGE_SIZE)
    # results is a list (one per batch or per image). We take first
    r = results[0]

    # Convert to supervision Detections
    detections = sv.Detections.from_ultralytics(r).with_nms()

    # Annotate
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)
    label_annotator = sv.LabelAnnotator()

    annotated = img.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    if show:
        cv2.imshow("Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, annotated)

    # Return detected class names and bounding boxes
    names = detections.class_id
    boxes = detections.xyxy  # list of [x1, y1, x2, y2]
    confidences = detections.confidence.tolist()
    return names, boxes, confidences

# Main Routine for Directory or Webcam
def run_on_folder(model, folder_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not fpath.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        outp = os.path.join(output_folder, fname)
        infer_image(model, fpath, show=False, save_path=outp)
        print("Processed:", fname)

def run_on_webcam(model, cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Cannot open webcam", cam_id)
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, size=IMAGE_SIZE)
        r = results[0]
        detections = sv.Detections.from_ultralytics(r).with_nms()

        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)
        label_annotator = sv.LabelAnnotator()

        annotated = frame.copy()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)

        cv2.imshow("Webcam Waste Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry Point

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Waste Segregation using YOLOv12")
    parser.add_argument("--mode", type=str, choices=["image", "folder", "webcam"], default="image")
    parser.add_argument("--source", type=str, default="test.jpg",
                        help="path to image, or folder if mode=folder")
    parser.add_argument("--output", type=str, default="out.jpg")
    args = parser.parse_args()

    model = load_model()

    if args.mode == "image":
        infer_image(model, args.source, show=True, save_path=args.output)
    elif args.mode == "folder":
        run_on_folder(model, args.source, output_folder=args.output)
    elif args.mode == "webcam":
        run_on_webcam(model, cam_id=int(args.source))
    else:
        print("Unknown mode", args.mode)
