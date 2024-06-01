import cv2
from ultralytics import YOLO
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
import time
import math


def predict(video_path, model_path):
    model = YOLO(model_path) # specify the model you would like to use
    print(model.names)
    video_path = video_path #make 0 if using webcam
    cap = cv2.VideoCapture(video_path)
    
    #INITIALIZING OCCUPACY GRID STARTING WITH BLANK FRAME OF ALL 0
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # occupancy_grid = np.zeros((image_height, image_width))
    
    #INITALIZING BUFFER
    memory_buffer = np.full((image_height, image_width), 0).astype(np.uint8) #stack with past frames of driveable area (intialize with full)
    print(memory_buffer)
    time_of_buffer = 0
    buffer_area = 0
    time_of_flip = time.time()
    cur_time = time.time()
    while cap.isOpened() and cv2.waitKey(1) & 0xFF != ord("q"): 
        success, frame = cap.read() # frame is (height, width, channels)
        
        
        
        if success:
            results = model.predict(frame, conf=0.7)[0] # this makes a prediction on a single frame of video
            lane_annotated_frame = results.plot()
            
            occupancy_grid = np.zeros((frame.shape[0], frame.shape[1]))
            #need to check if there exists a driveable mask
            if results.boxes is not None:
                boxes = results.boxes.xyxy.tolist()
                labels = results.boxes.cls.tolist()
                masks = results.masks.xy
                if 1.0 in labels:
                    for mask, label in zip(results.masks.xy, labels):
                        if label == 1.0:
                            #GET MASKS AND PLOTTING THEM ON OCCUPANCY GRID
                            segment_array = np.array([mask], dtype=np.int32) #lists all the points that make up the segmentation mask
                            cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 0, 0)) # connects all the points and fills in the space between them to make the mask on our occupancy grid
                            
                            
                            #CHECKING FOR SWITCHED EDGE CASE
                            current_time = time.time()
                            difference = current_time - time_of_flip
                            
                            switch = np.sum(np.logical_and(memory_buffer, np.logical_not(occupancy_grid)))/(np.clip(np.sum(memory_buffer)/255, 1, 100000))
                            print(switch)
                            print(difference)
                            if (switch >= 0.8 and difference < 4): #if we are still suspecting a switch and it hasnt been that much time
                                occupancy_grid = memory_buffer
                                time_of_flip = time.time()
                                print("FLIPPED")
                            elif switch >= 0.8 and difference < 8: #if we are still suspecting a switch but it has been too much time to be confident in the buffer
                                occupancy_grid.fill(255)
                                
                            else:
                                memory_buffer = occupancy_grid # if we get good output, add the most recent grid as a memory buffer
                            buffer_area = np.sum(occupancy_grid)//255
                            time_of_buffer = time.time()
                else: # if no detections are made we can use past detections or a fully filled grid as output
                    current_time = time.time()
                    buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)# between 1 and 1/e
                    if current_time - time_of_buffer < buffer_time: 
                        occupancy_grid = memory_buffer
                        print("BUFFER USED")
                    else:
                        occupancy_grid.fill(255)
                        print("FULL OCCUPANCY GRID USED")
                
                for i in range(occupancy_grid.shape[1]):
                        if np.any(occupancy_grid[-200:, i]):
                            occupancy_grid[-50:, i] = 255 
                            
                            
                if 0.0 in labels:
                    for box, label in zip(boxes, labels):
                        if(label == 0.0):
                            x_min, y_min, x_max, y_max = box
                            vertices = np.array([[x_min, y_min], 
                                                [x_max, y_min], 
                                                [x_max, y_max], 
                                                [x_min, y_max]], dtype=np.int32)
                            cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))  
                            
                    #FILLING IN THE BOTTOM TO MAKE THE DETECTIONGS FLUSH WITH THE BOTTOM OF THE GRID
                  
                
            
            else: # if no detections are made we can use past detections or a fully filled grid as output
                current_time = time.time()
                buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)# between 1 and 1/e
                if current_time - time_of_buffer < buffer_time: 
                    occupancy_grid = memory_buffer
                    print("BUFFER USED")
                else:
                    occupancy_grid.fill(255)
                    print("FULL OCCUPANCY GRID USED")
                    
                
            ####################### RESIZING FOR EASIER VIEWING ###########################
            occupancy_grid = cv2.resize(occupancy_grid, (image_height//2, image_width//2))
            frame = cv2.resize(frame, (image_height//2,image_width//2))
            lane_annotated_frame = cv2.resize(lane_annotated_frame, (image_height//2,image_width//2))
            ###############################################################################

            cv2.imshow("Lane Lines", occupancy_grid)
            cv2.imshow("frame", frame)
            cv2.imshow("YOLOv8 Inference", lane_annotated_frame)
            cv2.waitKey(10000)
            if cv2.waitKey(1) & 0xFF == ord("q"): # press q to quit the program
                break
        else:
            break #stop loop when video has ended
        
    ##########################
    # Must have these dont touch
    ##########################
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Check if the script is run as the main program
    if len(sys.argv) < 3:
        print("\nNot enough parameters!! Please enter 1 of the 3:\n ")
        print("1) python3 yolov8.py <video_path> <combined model>")
        print("2) python3 yolov8.py <video_path> <combined model>")
        print("3) enter 0 for the <video_path> if you are using a webcam device\n")
        sys.exit(1)

    # Extract the command line argument (parameter)
    parameter_value = sys.argv[1]
    model_name = sys.argv[2]
    
    if(parameter_value == "0"):
        parameter_value = int(parameter_value)

    # Call your function with the provided parameter
    predict(parameter_value, model_name)