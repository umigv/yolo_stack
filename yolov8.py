import cv2
from ultralytics import YOLO

model = YOLO('yolov8.pt')

video_path = "comp23_2.mp4" #make 0 if using webcam
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame)
        
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): #
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()







# from ultralytics import YOLO
# import cv2
# import math 
# # start webcam
# cap = cv2.VideoCapture("output2.mp4")
# cap.set(3, 480)#cap.set(3, 640)
# cap.set(4, 320)#cap.set(4, 480)

# # model
# model = YOLO("yolov8withpotholes.pt")


# while True:
#     success, img = cap.read()
#     results = model(img)

#     # coordinates
#     for r in results:
#         masks = r.masks

#         print(masks)

#         boxes = r.boxes

#         for box in boxes:
#             # bounding box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

#             # put box in cam
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # confidence
#             confidence = math.ceil((box.conf[0]*100))/100
#             #print("Confidence --->",confidence)

#             # class name
#             cls = int(box.cls[0])
#             #print("Class name -->", classNames[cls])

#             # object details
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 2

#             #cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

#     cv2.imshow('Video', img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()