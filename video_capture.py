'''Capture faces from computer webcam. Use OpenCV for each frame of video. '''

import cv2, time, pandas
from datetime import datetime

#captures first frame to compare subsequent frames with
first_frame = None
#list that tells when an object enters/exits the camera field of view
status_list = [None, None]
#list with times new object is detected entering or exiting
times = []
df = pandas.DataFrame(columns = ["Start","End"])

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    #status changes to 1 if detects object in camera field of view
    status = 0
    #Converts to grayscale & applies Gaus Blur to smoth edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    #if statement stores grayscale first fram to compare to
    if first_frame is None:
        first_frame = gray
        continue

    #stores grayscale, alpha channel, and threshold for difference
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 55, 255, cv2.THRESH_BINARY)[1]
    #Number of times to go through image & remove extraneous noise
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=5)

    #contours help clean up the Alpha channel so it can read the image more easily
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #draws box if item in that enters camera is over 10000 pixels in size  10000 pixels is a 100 x 100 pixel object
    for contour in cnts:
        if cv2.contourArea(contour) < 5000:
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    #This sets the time that an object enters or leaves the image
    status_list.append(status)
    if status_list [-1] == 1 and status_list [-2] == 0:
        times.append(datetime.now())
    if status_list [-1] == 0 and status_list [-2] == 1:
        times.append(datetime.now())    


    #Opens windows for display
    cv2.imshow("capture", gray)
    cv2.imshow("Delta", delta_frame)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("Color Frame", frame)


    key = cv2.waitKey(1)
    #print (gray)
    #print (delta_frame)

    #stops program by pressing "q"
    #If an object is in the frame when 1 is pressed, it logs the time 1 was pressed in the times list
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print (status_list)
print (times)

#iterates through list to append start/stop times to dataframe
for i in range (0, len(times), 2):
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index = True)

df.to_csv("Times.csv")

cam.release()
cv2.destroyAllWindows()