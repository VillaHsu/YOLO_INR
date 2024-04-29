from ultralytics import YOLO

model = YOLO('runs/detect/INR_64_stride256/weights/best.pt')
results = model.predict('./datasets/valid/images/EMPIAR-10057_7_new_jpeg.rf.3caa69bfd985580c9a94025ca4643901.jpg', 
	save=True, show=True, show_labels=False, conf=0.2, line_width=1)