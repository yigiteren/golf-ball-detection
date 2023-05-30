from asyncio import Queue
import asyncio
import math
from multiprocessing import Process
import os
import time
import cv2
import numpy as np
import serial
from ultralytics import YOLO

firstPoint = (100, 100)
secondPoint = (300, 100)
pixelLength = 0

async def send_data_async(serial_port, data):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, serial_port.write, data)
    await loop.run_in_executor(None, serial_port.flush)

def click_event(event, x, y, flags, param):

    global firstPoint
    global secondPoint
    global pixelLength

    if event == cv2.EVENT_LBUTTONDOWN: firstPoint = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN: secondPoint = (x, y)
    pixelLength = 300 / math.hypot(secondPoint[0] - firstPoint[0], secondPoint[1] - firstPoint[1])

def move_point_towards(point1, point2, d):
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Compute the vector from point2 to point1
    vector = point1 - point2
    
    # Compute the length of the vector
    length = np.linalg.norm(vector)
    
    if length == 0:
        raise ValueError("The points are the same.")
    elif length <= d:
        return tuple(map(int, point1))  # If the distance to move is greater than or equal to the distance between points, return point1
    
    # Scale the vector by d/length
    scaled_vector = vector * d / length
    
    # Add the scaled vector to point2
    moved_point = point2 + scaled_vector

    # Round to nearest integer and convert to int
    moved_point = np.rint(moved_point).astype(int)
    
    return tuple(moved_point)
        
def train_model():
    model = YOLO("C:\\Users\\yigit\\Desktop\\Golf Ball Finder\\yolov8n.pt")

    if __name__ == '__main__':
        model.train(data="C:\\Users\\yigit\\Desktop\\Golf Ball Finder\\data.yaml", epochs=3)
        metrics = model.val()  # evaluate model performance on the validation set
        success = model.export(format="onnx")  # export the model to ONNX format

def predict_webcam():

    global ser;

    model = YOLO(os.path.dirname(__file__) + "\\model.pt")
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 60)

    cv2.namedWindow('Golf Ball Finder')
    cv2.setMouseCallback('Golf Ball Finder', click_event)

    start_time = time.time()
    display_time = 0.1
    uc = 0
    fc = 0
    FPS = 0

    while capture.isOpened():

        read_start_time = time.time()
        success, frame = capture.read()
        #print("Frame read took: " + str((time.time() - read_start_time) * 1000) + " ms.")

        fps_start_time = time.time()

        uc+=1
        fc+=1
        TIME = time.time() - start_time

        if (TIME) >= display_time :
            FPS = fc / (TIME)
            fc = 0
            start_time = time.time()

        fps_disp = "FPS: "+str(FPS)[:5]

        cv2.putText(frame, fps_disp, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #print("FPS drawing took: " + str((time.time() - fps_start_time) * 1000) + " ms.")

        if success:
            prediction_start_time = time.time()
            results = model.predict(frame, verbose=False, half=True)
            #print("Prediction took: " + str((time.time() - prediction_start_time) * 1000) + " ms.")
            frame = cv2.circle(frame, firstPoint, 10, (255, 0, 0), 2)
            frame = cv2.circle(frame, secondPoint, 10, (0, 255, 0), 2)
            bounding_box_start_time = time.time()
            if len(results[0].boxes) > 0:
                b = results[0].boxes[0].xyxy.cpu().numpy()[0]
                ballRadius = int(((b[2] - b[0]) + (b[3] - b[1])) / 4)
                ballCenter = (int(round((b[0] + b[2]) / 2)), int(round((b[1] + b[3]) / 2)))
                ballPoint = move_point_towards(firstPoint, ballCenter, ballRadius) 
                frame = cv2.line(frame, firstPoint, ballPoint, (255, 0, 0), 2)
                frame = cv2.circle(frame, ballCenter, ballRadius, (0, 255, 255), 2)
                distance = math.hypot(ballPoint[0] - firstPoint[0], ballPoint[1] - firstPoint[1]) * pixelLength
                distance_str = "Distance: {:.2f}".format(distance)
                midpoint = ((firstPoint[0] + int(b[0])) // 2, (firstPoint[1] + int(b[1])) // 2)
                text_position = (midpoint[0] - len(distance_str) * 4, midpoint[1])
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 0)
                thickness = 1
                frame = cv2.putText(frame, distance_str, text_position, font, font_scale, font_color, thickness, cv2.LINE_AA)
                ser.write((str(int(distance)) + '\n').encode('utf-8'))
                ser.flush()

            #print("Bounding box took: " + str((time.time() - bounding_box_start_time) * 1000) + " ms.")

            imshow_box_start_time = time.time()
            cv2.imshow("Golf Ball Finder", frame)
            #print("cv2.imshow took: " + str((time.time() - imshow_box_start_time) * 1000) + " ms.")
            
            wait_key_start_time = time.time()

            if uc%5==0:
                cv2.waitKey(1)
                uc = 0

            #print("cv2.waitkey took: " + str((time.time() - wait_key_start_time) * 1000) + " ms.")
            #print("Whole frame took: " + str((time.time() - read_start_time) * 1000) + " ms.\n")
        else:
            break

    capture.release()
    cv2.destroyAllWindows()

ser = serial.Serial('COM6', 115200)
time.sleep(2) # Small wait time to make sure that serial connection has time to initialize.

#train_model()
predict_webcam()