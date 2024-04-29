from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path

# Download YOLOv8 model
yolov8_model_path = "runs/detect/CNL_single/weights/best.pt"
#download_yolov8s_model(yolov8_model_path)

predict(
    model_type="yolov8",
    model_path=yolov8_model_path,
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="datasets_cavity/valid/images",
    slice_height=128,
    slice_width=128,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)