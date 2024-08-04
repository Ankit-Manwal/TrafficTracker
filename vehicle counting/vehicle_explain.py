import cv2
import numpy as np

# Define the vertical position of the counting line
count_line_position = 550

# Function to calculate the center of a rectangle
def centre_of_rectangle(x, y, w, h):
    return x + int(w / 2), y + int(h / 2)

# Initialize variables
detect = []
offset = 6  # Tolerance for object position relative to the counting line
counter = 0

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

    # Dilate the subtracted image to fill in holes and np.ones((5,5)) creates a 5x5 matrix of ones, which serves as the structuring element.
    dilate = cv2.dilate(img_subt, kernel=np.ones((5,5), np.uint8), iterations=1 ) #iterations=1 by default and img_subt=binary

    # Create a structuring element (kernel) for morphological operations
    '''cv2.MORPH_ELLIPSE: Specifies that the structuring element should be elliptical in shape. 
    This is useful for operations where a more rounded or circular neighborhood is desired compared to a rectangular one.'''
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))

    # Apply morphological closing (dilation followed by erosion) to remove noise
    '''erosion=Shrinks the white regions in the binary image. 
     It reduces the size of the objects by removing pixels from the boundaries) to remove noise'''
    dilatada = cv2.morphologyEx(src=dilate, op=cv2.MORPH_CLOSE, kernel=kernel)
    dilatada = cv2.morphologyEx(src=dilatada, op= cv2.MORPH_CLOSE, kernel=kernel)
 
    #draw line in frame
    '''pt =coordinate'''
    cv2.line(img=frame1, pt1=(25,count_line_position), pt2=(1200,count_line_position) ,color=(255,127,0),  thickness=3 )


   # Find contours of the objects detected
    '''The cv2.findContours() function in OpenCV is used to detect and retrieve contours
      from a binary image. Contours are curves joining all the continuous points along a 
      boundary with the same color or intensity. This function is often used in image processing 
      for object detection, shape analysis, and feature extraction.'''
    #cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments and leaves only their end points. This saves memory.
    #cv2.RETR_TREE: Retrieves all contours and creates a full hierarchy of nested contours.


    #Find contours in the dilated image
    contours, _ = cv2.findContours(image=dilatada,  mode= cv2.RETR_TREE,   method=cv2.CHAIN_APPROX_SIMPLE)

    #Draw the contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > 2000:           # Minimum area to filter out noise
            ''' (w>=min_width_rect=80) and (h>=min_height_rect=80)'''
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img=frame1, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            
            cv2.putText(img=frame1,text="Vehicle:" + str(counter), org=(x, y-20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255,244,0), thickness=2 )

            centre= centre_handle(x,y,w,h)
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



    # Display the dilated image with detected objects
    cv2.imshow('Detector', blur)

    # Display the original video frame
    cv2.imshow('Video Original', frame1)

    # Break the loop if the 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()
