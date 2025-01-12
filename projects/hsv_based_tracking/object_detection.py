import numpy as np
import cv2 as cv

capture = cv.VideoCapture('/home/shivam/computerVision/projects/hsv_based_tracking/Football in full flow.mp4')

# defining color ranges
blue_lower = np.array([85, 50, 100])  # Lower range for blue
blue_upper = np.array([110, 255, 255])  # Upper range for blue

black_lower = np.array([0, 0, 0])  
black_upper = np.array([180, 255, 50]) 

# cropping the video to reduce audience noise
#crop = 200


def detectPlayers(frame,color_range_lower,color_range_upper):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv,color_range_lower,color_range_upper)


    # defining roi or region of interest
    mask_exclude = [(18,0),(383,223)]
    goal_mask = np.zeros(mask.shape,np.uint8)
    cv.rectangle(goal_mask,mask_exclude[0],mask_exclude[1],255,-1)
    mask = cv.bitwise_and(mask,cv.bitwise_not(goal_mask))

    # morphological operation to reduce noise
    kernel = np.ones((3,3),np.uint8)
    mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel,iterations=3)
    mask = cv.dilate(mask,kernel,iterations=3)
    cv.imshow('mask',mask)
    contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    players = []
    # filtering contours by area
    for contour in contours:
        if cv.contourArea(contour) > 450:
            x,y,w,h = cv.boundingRect(contour)
            players.append((x,y,w,h))

    return players

while True:
    isTrue,frame = capture.read()
    
    if not isTrue:  
        print("Failed to grab a frame")
        break
    
    # cropping the frame
    #frame = frame[crop:,:]

    blue_team = detectPlayers(frame,blue_lower,blue_upper)
    black_team = detectPlayers(frame,black_lower,black_upper)

    # creating rectangles around detected players
    for x,y,w,h in blue_team:
        cv.rectangle(frame,(x,y),(x+w+2, y+h+10),(255,0,0),2)
    
    for x,y,w,h in black_team:
        cv.rectangle(frame,(x,y),(x+w+2, y+h+5),(0,255,0),2)

    cv.imshow('Player detection',frame)

    # break on ESC
    if cv.waitKey(20) & 0xFF == 27:   
        break
        
capture.release()
cv.destroyAllWindows()