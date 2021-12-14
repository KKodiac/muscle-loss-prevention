# 파넬백 광학 흐름 변환으로 영상을 만들어주는 파일

import cv2
import numpy as np

cap = cv2.VideoCapture("./data/squat_wrong1.mp4")

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('./data/optical_squat_wrong1.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
i = 0
prediction_str = ""
repetitions = 0
up = 0
down = 0
no_move = 0
current_move = 0
initial = -1
while (cap.isOpened()):
    i += 1

    ret, frame2 = cap.read()
    if not (ret): break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Mediapipe Feed', rgb)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    out.write(rgb)
    prvs = next

print("Video Generated")
out.release()
cap.release()
cv2.destroyAllWindows()