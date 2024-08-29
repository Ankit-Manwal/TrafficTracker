from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import sort_modified as sort
import time

# Define COCO classes
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Load YOLO model
model = YOLO('yolo_weights/yolov8l.pt')

# Tracking
tracker = sort.Sort(max_age=2, min_hits=3, iou_threshold=0.3)

# Capture video
cap = cv2.VideoCapture('images and videos/video3.mp4')
cap.set(3, 1280)
cap.set(4, 720)

# Load and resize mask
mask = cv2.imread('images and videos/mask3.png')
mask = cv2.resize(mask, (1280, 720))

# Line coordinate at which vehicles are counted
countLine_coordinate = [(0, 550), (1280, 550)]
offset = 20

pixels_per_meter=12

#Valid speed calculating area w.r.t pixels_per_meter
speedline_coordinate1 = [(0, 300), (1280, 300)]
speedline_coordinate2 = [(0, 500), (1280, 500)]

# Initialize variables for counting and speed calculation
totalCount = []
vehicle_positions = {}


prev_time = time.time()




def estimate_speed( vehicle_positions, ID ):
    if ID in vehicle_positions and (cy>=speedline_coordinate1[0][1] and cy<=speedline_coordinate2[0][1]):
        prev_cx, prev_cy, prev_time = vehicle_positions[ID]
        
        # Calculate distance in pixels
        distance_pixels = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        
        # Convert distance from pixels to meters
        distance_meters = distance_pixels / pixels_per_meter
        
        # Time between frames in seconds
        time_elapsed = 1 / fps_display
        
        # Speed in meters per second
        speed_mps = distance_meters / time_elapsed
        
        # Convert speed to kilometers per hour
        speed_kph = speed_mps * 3.6
        return speed_kph
    else:
        return 0



while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.resize(frame, (1280, 720))
    masked_frame = cv2.bitwise_and(frame, mask)  # overlap mask and actual video to get only counting area

    # imgGraphics= cv2.imread("graphic,png", cv2.IMREAD_UNCHANGED)     #for pasting a image over video or screen
    # img=cvzone.overlayPNG(img, imgGraphics,(0,0))

    results = model(masked_frame, stream=True)
    detections=np.empty(shape=(0, 6)) # initially 0  & each row 5 elements [x1, y1, x2, y2, confidence]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  #  tensor(794.) tensor(471.) tensor(854.) tensor(519.)

            # Confidence score
            confidence = float(box.conf[0])
            confidence = np.round(confidence * 100, 2)

            # Class name
            curr_class_index = int(box.cls[0])
            curr_class = coco_classes[curr_class_index]
            
            
            if (curr_class in ["car", "truck", "bus"]) and confidence > 30:
                currArray = np.array([x1, y1, x2, y2, confidence, curr_class_index])
                detections = np.vstack((detections, currArray))



    # Draw the counting line
    cv2.line(img=frame, pt1=countLine_coordinate[0], pt2=countLine_coordinate[1], color=(0, 127, 255), thickness=3)

    # Draw the speed line
    cv2.line(img=frame, pt1=speedline_coordinate1[0], pt2=speedline_coordinate1[1], color= (0, 255, 255), thickness=1)
    cv2.line(img=frame, pt1=speedline_coordinate2[0], pt2=speedline_coordinate2[1], color=(0, 127, 255), thickness=1)

    # Tracking vehicles
    resultsTracker = tracker.update(detections)


    # Calculate and display the FPS
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img=frame, text=f'FPS: {int(fps_display)}', org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    

    for result in resultsTracker:
        x1, y1, x2, y2, ID ,cls= map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        

        # calculate speed
        speed_kph= estimate_speed( vehicle_positions, ID )

        if speed_kph!=0:
            cvzone.putTextRect(img=frame, text=f'{ID} {coco_classes[cls]} {int(speed_kph)} km/h', pos=(max(0, x1), max(35, y1 - 20)),
                               thickness=2, scale=1, offset=0)
        else:
            cvzone.putTextRect(img=frame, text=f'{ID} {coco_classes[cls]}', pos=(max(0, x1), max(35, y1 - 20)),
                               thickness=2, scale=1, offset=0)

        
        #delete vehicle location if it passed through speed calculating area
        if ID in vehicle_positions and cy>speedline_coordinate2[0][1]:
            del vehicle_positions[ID]

        # Update vehicle positions
        vehicle_positions[ID] = (cx, cy, time.time())

        # Draw bounding box and ID
        cvzone.cornerRect(frame, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))  # l length  of corner green part, rt reactangle thickness
        cv2.circle(img=frame, center=(cx, cy), radius=4, color=(0, 0, 255), thickness=-1)  #shift=cv2.FILLED) for filling circle if thickness not =-1


        # Count vehicles crossing the line
        if (550 - offset < cy < 550 + offset) and ID not in totalCount:
            totalCount.append(ID)
            cv2.line(img=frame, pt1=countLine_coordinate[0], pt2=countLine_coordinate[1], color=(0, 255, 0), thickness=3)


    # Display the total count
    cv2.putText(img=frame, text="Vehicle counter:" + str(len(totalCount)), org=(450, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5)



    # Show the frame
    cv2.imshow("image", frame)

    # Break the loop if the 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


     #def update(self, dets=np.empty((0, 5))):
"""
Params:
    dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
Returns the a similar array, where the last column is the object ID.

NOTE: The number of objects returned may differ from the number of detections provided.

 ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
"""