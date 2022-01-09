import cv2
import cvzone
from cvzone.PlotModule import LivePlot
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
#        for id in idList:
#            cv2.circle(img, face[id], 3, (255, 0, 255), cv2.FILLED)
        leftUpEye = face[159]
        leftDownEye = face[23]
        leftLeftEye = face[130]
        leftRightEye = face[243]

        lengthVer, _ = detector.findDistance(leftDownEye, leftUpEye)
        lengthHor, _ = detector.findDistance(leftLeftEye, leftRightEye)
#        cv2.line(img, leftUpEye, leftDownEye, (0, 200, 0))
#        cv2.line(img, leftLeftEye, leftRightEye, (0, 200, 0))

        ratio = (lengthVer / lengthHor) * 100
        ratioList.append(ratio)

        if len(ratioList) > 3:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 33:
            imgPlot = plotY.update(ratioAvg, (0, 200, 0))
        else:
            imgPlot = plotY.update(ratioAvg, (255, 0, 255))

        print(ratioAvg)
        img = cv2.resize(img, (640, 360))
        imageStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imageStack = cvzone.stackImages([img, ], 2, 1)

    cv2.imshow("Image", imageStack)
    cv2.waitKey(40)
