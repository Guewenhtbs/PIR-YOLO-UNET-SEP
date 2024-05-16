from ultralytics import YOLO #use pip install ultralytics==8.0.38
from pathlib import Path

model = YOLO("yolov8n-seg.pt")

results = model.train(
        batch=8,
        device="cpu",
        data= "data.yaml",
        epochs=20,
        imgsz=256,
        single_cls=True,
    )