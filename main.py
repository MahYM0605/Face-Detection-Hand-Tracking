import cv2 as cv
import mediapipe as mp
import time as t

class FaceDetection():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self,frame,draw = True):
        
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        
        if self.results.detections:
            for id, detection in  enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw ,ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    frame = self.fancyDraw(frame, bbox)
                
                    cv.putText(frame,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    
        return frame, bboxs

    def fancyDraw(self, frame, bbox, l = 50, t =5, rt = 1):
        x,y,w,h = bbox
        x1, y1 = x+w,y+h
        
        cv.rectangle(frame,bbox,(255,0,255),rt)
        
        # Top Left x,y
        cv.line(frame, (x,y), (x+l,y),(255,0,255), t)
        cv.line(frame, (x,y), (x,y+l),(255,0,255), t)
        
        # Top Right x1,y
        cv.line(frame, (x1,y), (x1-l,y),(255,0,255), t)
        cv.line(frame, (x1,y), (x1,y+l),(255,0,255), t)
        
        # Bottom Left x,y1
        cv.line(frame, (x,y1), (x+l,y1),(255,0,255), t)
        cv.line(frame, (x,y1), (x,y1-l),(255,0,255), t)
        
        # Bottom Right x1,y1
        cv.line(frame, (x1,y1), (x1-l,y1),(255,0,255), t)
        cv.line(frame, (x1,y1), (x1,y1-l),(255,0,255), t)
        
        return frame
    
class handDetector():
    def __init__(self, mode=False,maxHands=2, model_complexity = 1,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame
    
    def findPosition(self,img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int (lm.y*h)
                # print(id,cx,cy)
                lmList.append([id,cx,cy],)
                if draw:
                    cv.circle(img, (cx,cy), 5, (255,0,255),cv.FILLED)
        return lmList
    
def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    hdetector = handDetector()
    fdetector = FaceDetection()
    
    while True:
        success, frame = cap.read()
        frame =  hdetector.findHands(frame)
        lmList = hdetector.findPosition(frame,draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        frame, bboxs = fdetector.findFaces(frame)
        print(bboxs)
        
        cTime = t.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame,f'FPS: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        cv.imshow('WebCam', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
if __name__ == "__main__":
    main()