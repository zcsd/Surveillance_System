from face_detector import FaceDetector
import cv2
import imutils
import argparse

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Initialize face detector
face_detector = FaceDetector()

frame = cv2.imread("images/3.jpg", 1)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# (top, right, bottom, left) 
known_face_locs = face_detector.detect(frame)

top = known_face_locs[0][0]
bottom = known_face_locs[0][2]
left = known_face_locs[0][3]
right = known_face_locs[0][1]

cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

frame_roi = frame_gray[top:bottom, left:right]
variance = variance_of_laplacian(frame_roi)
print(variance)

cv2.putText(frame, str("%.2f" % variance), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
