# importing necessary libraries
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# importing the cascade file into the variable 'face_cascade'
face_cascade = cv2.CascadeClassifier(r"C:\Users\kirti\haarcascade_frontalface_default.xml")

# importing the image into the variable 'img'.
img = cv2.imread(r"C:\Users\kirti\lena.png")

# Reading the image and converting it to RGB and storing it in the variable 'gray'.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Now, if the faces are detected, the positions of detected faces is returned as Rect(x,y,w,h) and ROI can be created
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# returning the co-ordinates of the ROI.
print(faces)

# drawing a rectangle on the ROI.
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (31, 79, 254), 2)

# rendering the video with effective facetracking to the console by using the function '.imshow()'.
cv2.imshow('Face Detection on Image using Mediapipe', img)

# setting up '.waitkey()' to wait for a specific time until any key is pressed and break the loop.
cv2.waitKey(0)
# Destroying all windows.
cv2.destroyAllWindows()

# Creating a variable to store the video using the '.VideoCapture()' function.
cap = cv2.VideoCapture(r"C:\Users\kirti\Bill_Gates.mp4")

# Specifying the detection and tracking confidence using 'mp_pose.pose()'.
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        # Capturing the video frame by frame using the '.read()' method.
        success, image = cap.read()

        # To improve performance, marking the image as not writable and passing by reference.
        image.flags.writeable = False
        # Reading the frames and converting them to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Detecting faces in the frame using the function 'face_detection.process()'.
        results = face_detection.process(image)

        image.flags.writeable = True

        # Reading the frames and converting them to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If a face is detected we draw a rectangle on the ROI.
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        cv2.imshow('Face Detection on Video using Mediapipe', image)
        # Setting 'x' as the quitting button.
        if cv2.waitKey(5) & 0xFF == ord('x'):
            break
cap.release()
