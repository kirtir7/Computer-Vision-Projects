# Importing necessary modules.
import cv2
import time

video_cap = cv2.VideoCapture(r'C:\Users\kirti\Bill_Gates.mp4')
time_1 = time.time()
time_2 = time.time()
while 1:
    res, frame = video_cap.read()
    time_3 = time.time()
    frames_per_sec = 1 / (time_3 - time_1)
    time_1 = time_3
    cv2.putText(frame, f'FPS: {int(frames_per_sec)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 0, 0), 5)
    if res:
        cv2.imshow("Output Video with FPS", frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    else:
        break
video_cap.release()
cv2.destroyAllWindows()
