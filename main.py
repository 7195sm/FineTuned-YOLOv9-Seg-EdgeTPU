import cv2
import numpy as np
import pandas as pd
import cvzone
from ultralytics import YOLO
import time

#load model
model = YOLO("best_full_integer_quant_edgetpu.tflite", task="segment")

cap = cv2.VideoCapture(0)

my_file = open("butterfly.txt","r")
data = my_file.read()
class_list = data.split("\n")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (1020, 600))
    overlay = frame.copy()


    results = model.predict(frame, imgsz=256)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
    
    # Overlay segmentation mask on the original frame
    for result in results:
        if result.masks is not None:
           for j, mask in enumerate(result.masks.data):
               mask = mask.numpy() * 255
               mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
               mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
               contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               cv2.fillPoly(mask_bgr, contours, (0, 0, 255))
               frame = cv2.addWeighted(frame, 1, mask_bgr, 0.5, 0)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    cvzone.putTextRect(frame, f'FPS: {round(fps,2)}', (10,30), 1, 1)

    cv2.imshow("YOLOv9 Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


