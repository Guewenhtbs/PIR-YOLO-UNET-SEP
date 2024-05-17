#!pip install --upgrade ultralytics
from ultralytics import YOLO

model1 = YOLO("yolov8n-seg.pt")

results1= model1.train(
            batch = 10,
            device = 0,
            data = "data1.yaml",
            epochs = 200,
            imgsz = 640,
            cos_lr = True,
            single_cls = True,
            translate = 0.3,
            degrees = 90,
            flipud = 0.3,
            scale = 0.8,
)