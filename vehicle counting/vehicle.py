import cv2
import numpy as np


count_line_position=550

def centre_of_rectangle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx, cy


detect=[]
offset=6 # allowable error b/w pixel
counter=0

# Initialize video capture from a file named 'video.mp4'
cap = cv2.VideoCapture('video.mp4')


# Initialize the background subtractor using the MOG (Mixture of Gaussians) algorithm
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


# Infinite loop to process video frames until break
while True:
    # Read a frame from the video capture
    ret, frame1 = cap.read()
    
    # If frame is not read correctly, exit loop
    if not ret:
        break
    
    # Convert the frame to grayscale
    grey = cv2.cvtColor(src=frame1, code=cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur (background)to the grayscale image
    blur = cv2.GaussianBlur(src=grey, ksize=(3,3),sigmaX= 5)  #sigmaX= This controls the amount of blur applied in the horizontal direction

    # Apply the background subtractor on the blurred image
    img_subt = algo.apply(blur)

    # Dilate the subtracted image to fill in holes 
    dilate = cv2.dilate(img_subt, kernel=np.ones((5,5), np.uint8), iterations=1 ) 

    # Create a structuring element (kernel) for morphological operations
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))

    # Apply morphological closing (dilation followed by erosion) to remove noise
    dilatada = cv2.morphologyEx(src=dilate, op=cv2.MORPH_CLOSE, kernel=kernel)
    dilatada = cv2.morphologyEx(src=dilatada, op= cv2.MORPH_CLOSE, kernel=kernel)
 
    #draw line in frame
    cv2.line(img=frame1, pt1=(25,count_line_position), pt2=(1200,count_line_position) ,color=(255,127,0),  thickness=3 )


   # Find contours of the objects detected
    contours, _ = cv2.findContours(image=dilatada,  mode= cv2.RETR_TREE,   method=cv2.CHAIN_APPROX_SIMPLE)

    #Draw the contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > 2000:           # Minimum area to filter out noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img=frame1, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            
            cv2.putText(img=frame1,text="Vehicle:" + str(counter), org=(x, y-20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255,244,0), thickness=2 )

            centre= centre_of_rectangle(x,y,w,h)
            detect.append(centre)
            cv2.circle(img=frame1, center=centre, radius=4, color=(0,0,255), thickness=-1)
            
            
            for (x,y) in detect:
                '''Crossing Detection: Checks if the y coordinate of the vehicle's center is within the range 
                of the count_line_position plus/minus offset. If it is, it means the vehicle has crossed the line.'''

                if y < (count_line_position + offset) and y> (count_line_position - offset):
                    counter+=1
                
                    cv2.line(img=frame1, pt1=(25,count_line_position), pt2=(1200,count_line_position) ,color=(0,127,255),  thickness=3 )
            
                    detect.remove((x,y))
            
                    #Display Counter on console
                    print("Vehicle Counter:" + str(counter))
            
     #Display Counter on Frame   
    cv2.putText(img=frame1,text="Vehicle counter:" + str(counter), org=(450,70), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=2, color=(0,0,255), thickness=5 )



    cv2.imshow('Detector', blur)

    cv2.imshow('Video Original', frame1)

    # Break the loop if the 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()
