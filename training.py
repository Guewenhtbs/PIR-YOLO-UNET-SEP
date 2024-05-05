from ultralytics import YOLO #use pip install ultralytics==8.0.38
from pathlib import Path

model = YOLO("yolov8n-seg.pt")

results = model.train(
        batch=8,
        device="cpu",
        data= str(Path("data.yaml").resolve()),
        epochs=7,
        imgsz=256,
    )