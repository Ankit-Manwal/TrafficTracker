from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import sort_modified as sort
import time

# Define COCO classes, including vehicle classes for detection
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

# Load YOLOv8 model for object detection
model = YOLO('yolo_weights/yolov8l.pt')

# Initialize SORT tracker for tracking detected objects
tracker = sort.Sort(max_age=2, min_hits=3, iou_threshold=0.3)

# Capture video feed
cap = cv2.VideoCapture('images and videos/video3.mp4')
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Load and resize the mask to focus on a specific area of the frame
mask = cv2.imread('images and videos/mask3.png')
mask = cv2.resize(mask, (1280, 720))

# Line coordinates for counting vehicles
countLine_coordinate = [(0, 550), (1280, 550)]
offset = 20  # Offset for counting line

# Pixel-to-meter conversion factor (adjust according to real-world scenario)
pixels_per_meter = 12

# Coordinates for speed calculation zone
speedline_coordinate1 = [(0, 300), (1280, 300)]
speedline_coordinate2 = [(0, 500), (1280, 500)]

# Initialize variables for tracking vehicles and calculating speed
totalCount = []  # List to store IDs of counted vehicles
vehicle_positions = {}  # Dictionary to store positions of vehicles

prev_time = time.time()  # Variable to calculate FPS




def estimate_speed(vehicle_positions, ID):
    """Estimate the speed of a vehicle given its current and previous position."""
    if ID in vehicle_positions and (cy >= speedline_coordinate1[0][1] and cy <= speedline_coordinate2[0][1]):
        prev_cx, prev_cy, prev_time = vehicle_positions[ID]
        
        # Calculate distance traveled in pixels
        distance_pixels = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
        
        # Convert distance from pixels to meters
        distance_meters = distance_pixels / pixels_per_meter
        
        # Time elapsed since the last position update
        time_elapsed = 1 / fps_display
        
        # Calculate speed in meters per second
        speed_mps = distance_meters / time_elapsed
        
        # Convert speed to kilometers per hour
        speed_kph = speed_mps * 3.6
        return speed_kph
    else:
        return 0  # Return 0 if speed cannot be calculated


while True:
    success, frame = cap.read()  # Capture a frame from the video
    if not success:
        break  # Exit loop if no frame is captured
    
    frame = cv2.resize(frame, (1280, 720))  # Resize frame to match the mask
    masked_frame = cv2.bitwise_and(frame, mask)  # Apply the mask to focus on specific area

    # Get detections from YOLOv8 model
    results = model(masked_frame, stream=True)
    detections = np.empty(shape=(0, 6))  # Initialize an empty array for detections

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates and confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            confidence = np.round(confidence * 100, 2)

            # Get the class name and index
            curr_class_index = int(box.cls[0])
            curr_class = coco_classes[curr_class_index]
            
            # Filter detections for vehicles (car, truck, bus) with confidence > 30%
            if (curr_class in ["car", "truck", "bus"]) and confidence > 30:
                currArray = np.array([x1, y1, x2, y2, confidence, curr_class_index])
                detections = np.vstack((detections, currArray))

    # Draw the counting line on the frame
    cv2.line(img=frame, pt1=countLine_coordinate[0], pt2=countLine_coordinate[1], color=(0, 127, 255), thickness=3)

    # Draw the speed calculation lines
    cv2.line(img=frame, pt1=speedline_coordinate1[0], pt2=speedline_coordinate1[1], color=(0, 255, 255), thickness=1)
    cv2.line(img=frame, pt1=speedline_coordinate2[0], pt2=speedline_coordinate2[1], color=(0, 127, 255), thickness=1)

    # Update tracker with current detections
    resultsTracker = tracker.update(detections)


    # Calculate FPS for performance monitoring
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on the frame
    cv2.putText(img=frame, text=f'FPS: {int(fps_display)}', org=(20, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)


    for result in resultsTracker:
        x1, y1, x2, y2, ID, cls = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Estimate the speed of the vehicle
        speed_kph = estimate_speed(vehicle_positions, ID)

        # Display the vehicle ID and speed if available
        if speed_kph != 0:
            cvzone.putTextRect(img=frame, text=f'{ID} {coco_classes[cls]} {int(speed_kph)} km/h', pos=(max(0, x1), max(35, y1 - 20)),
                               thickness=2, scale=1, offset=0)
        else:
            cvzone.putTextRect(img=frame, text=f'{ID} {coco_classes[cls]}', pos=(max(0, x1), max(35, y1 - 20)),
                               thickness=2, scale=1, offset=0)


        # Remove vehicle from tracking if it passes the speed calculation zone
        if ID in vehicle_positions and cy > speedline_coordinate2[0][1]:
            del vehicle_positions[ID]


        # Update vehicle positions with the current position
        vehicle_positions[ID] = (cx, cy, time.time())

        # Draw the bounding box and ID on the frame
        cvzone.cornerRect(frame, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cv2.circle(img=frame, center=(cx, cy), radius=4, color=(0, 0, 255), thickness=-1)

        # Count vehicles that cross the counting line
        if (550 - offset < cy < 550 + offset) and ID not in totalCount:
            totalCount.append(ID)
            cv2.line(img=frame, pt1=countLine_coordinate[0], pt2=countLine_coordinate[1], color=(0, 255, 0), thickness=3)

    # Display the total count of vehicles
    cv2.putText(img=frame, text="Vehicle counter:" + str(len(totalCount)), org=(450, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5)

    # Show the processed frame
    cv2.imshow("image", frame)

    # Break the loop if the 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
