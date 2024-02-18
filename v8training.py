from ultralytics import YOLO

model = YOLO("yolov8_180epochswithpotholes.pt")

model.predict(source = "test1.mp4", show=True, save=True, hide_labels=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2)
