from .yolov7 import yolo
from .yolov7.yolo import YOLO


__all__ = ['build_detector']

def build_detector():
    return YOLO()
