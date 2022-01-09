import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, (255, 0, 255), cv2.FILLED)

#    img = cv2.resize(img, (640, 360))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
