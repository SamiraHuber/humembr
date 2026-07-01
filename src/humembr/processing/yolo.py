from ultralytics import YOLO  # type: ignore


def get_yolo_model():
    yolo_model = YOLO("yolo11m-pose")
    return yolo_model
