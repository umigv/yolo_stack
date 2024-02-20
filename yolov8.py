import cv2
from ultralytics import YOLO
import numpy as np
import sys
import matplotlib.pyplot as plt
import json

def predict(video_path, lane_model, hole_model=None):

    lane_model = YOLO(lane_model) # specify the model you would like to use
    hole_model = YOLO(hole_model) if hole_model is not None else None

    video_path = video_path #make 0 if using webcam
    cap = cv2.VideoCapture(video_path)


    while cap.isOpened() and cv2.waitKey(1) & 0xFF != ord("q"):
        success, frame = cap.read()
        
        if success:
            # print("here1")
            lane_results = lane_model(frame) # this makes a prediction on a single frame of video
            hole_results = hole_model(frame) if hole_model is not None else None
            lane_annotated_frame = lane_results[0].plot() # 
            hole_annotated_frame = hole_results[0].plot() if hole_model is not None else None
            # print(lane_results[0])
            
            # for results in results:
            #     masks = results.masks  # masks gives us the coordinates for drivable area
            #     print(masks)
            #     break
            # print(f"lane_results.shape: {len(lane_results)}")
            image_height = frame.shape[0]
            image_width = frame.shape[1]
            occupancy_grid = np.zeros((image_height, image_width))
            # print(len(lane_results))
            if len(lane_results) > 1:
                print("BING BONG")
            for r in lane_results:
                # r.show()
                # print(r.probs)
                jsons = json.loads(r.tojson())
                if(len(jsons) != 0):
                    confidence = jsons[0]['confidence']
                    print(confidence)
                # print(confidence)
                if r.masks is not None:
                    # print(r.masks.xy[0])
                    for segment in r.masks.xy:
                        print(r.boxes.conf)
                        if(len(segment) != 0):
                            segment_array = np.array([segment], dtype=np.int32)
                            
                            # print(f"image height: {image_height}")
                            # print(f"image width: {image_width}")
                            # print(segment[0])
                            cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 255, 255))
                        # print("here4b")
                        
            if hole_results is not None:
                for r in hole_results:
                    if r.boxes is not None:
                        # print("here4")
                        for segment in r.boxes.xyxyn:
                            x_min, y_min, x_max, y_max = segment
                            vertices = np.array([[x_min*image_width, y_min*image_height], 
                                                [x_max*image_width, y_min*image_height], 
                                                [x_max*image_width, y_max*image_height], 
                                                [x_min*image_width, y_max*image_height]], dtype=np.int32)
                            print(vertices)
                            cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))


            cv2.imshow("Lane Lines", occupancy_grid)
            cv2.imshow("YOLOv8 Inference", lane_annotated_frame)
            ##################For Nav Output not necessary for running
            # summed_grid = np.sum(occupancy_grid, axis=2)

            # BINARY GRID TO SEND TO NAV
            # binary_grid = np.array(np.where(summed_grid == 0, 0, 1))
            # print("here5")
            #################################
            # plt.imshow(binary_grid, cmap='binary_r')
            # plt.show()
            # if hole_model is not None:
            #     cv2.imshow("Potholes", hole_annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"): # press q to quit the program
                break
        else:
            break
        
    ##########################
    # Must have these dont touch
    ##########################
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Check if the script is run as the main program
    if len(sys.argv) < 3:
        print("Not enough parameters!! Please enter python3 yolov8.py <video_path> <model_name>")
        sys.exit(1)

    # Extract the command line argument (parameter)
    parameter_value = sys.argv[1]
    model_name = sys.argv[2]
    hole_model = None
    if len(sys.argv) == 4:
        hole_model = sys.argv[3]
    if(parameter_value == "0"):
        parameter_value = int(parameter_value)

    # Call your function with the provided parameter
    predict(parameter_value, model_name, hole_model)




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