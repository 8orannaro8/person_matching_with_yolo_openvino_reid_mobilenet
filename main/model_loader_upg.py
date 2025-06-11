# model_loader.py
from model_loader_openvino import OpenVINOYOLOv5
from sort_upg import Sort
from final_practical_reid import FinalPracticalReID

def load_yolo_model():
    return OpenVINOYOLOv5('yolov5n.xml', device='CPU', conf_threshold=0.25)

def load_tracker():
    return Sort()

def load_reid_model():
    return FinalPracticalReID()
