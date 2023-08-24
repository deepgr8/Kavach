from ultralytics import YOLO
model = YOLO('./runs/detect/Signtrain/weights/best.pt')
results = model.predict(source=0,show=True)