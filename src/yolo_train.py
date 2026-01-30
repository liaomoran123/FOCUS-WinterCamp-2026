from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco128.yaml", epochs=20, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model.predict(source="D:/_desktop/shujvji/", save=True)
