import cv2
import mediapipe as mp
import time

import cv2

cap=cv2.VideoCapture(0)
pTime=0
mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
#by modifying the detectionconfidence (increasing its value) the false positive are eliminated teh bounding box will be more focused on the face
faceDetection=mpFaceDetection.FaceDetection(0.75)
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #with the following two lines of code reduce the frames per second
    results=faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id,detection in enumerate(results.detections):
            #print(id,detection)
            #print(id,detection.score)
            #this is a very long call for a one value:print(detection.location_data.relative_bounding_box.xmin)
            #in that case we will store all of this in only one bounding box then extract the data 
            print(detection.location_data.relative_bounding_box)
            #the following boundix box comes from the class
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),int(bboxC.width*iw),int(bboxC.height*ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            #mpDraw.draw_detection(img,detection)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("IMAGE",img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break

cv2.destroyAllWindows()

    