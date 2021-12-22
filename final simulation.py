#기본모듈
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

#전처리 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#머신러닝 모델 라이브러리
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
import tensorflow as tf

#평가지표
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report

#사용할 영상처리 및 골격인식 라이브러리
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
LR = joblib.load('pushup.pkl')
LR2 = joblib.load('squat.pkl')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


model = tf.keras.models.load_model('model.h5')
cap = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# out = cv2.VideoWriter('my_count2.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

ex = 1
flag = False
flag2 = False
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    a = []
    b = []
    c = []
    d = []

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
    # Curl counter variables
    counter = 0
    stage = None
    while cap.isOpened():
        i += 1
        ret, frame2 = cap.read()
        if not ret: break
        if cv2.waitKey(30) >= 0: break
        # Recolor image to RGB
        image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image2 = cv2.resize(rgb, (64, 64))
        image2 = image2.reshape((1,) + image2.shape)
        image2 = image2 / 255.0
        prediction = np.argmax(model.predict(image2), axis=-1)[0]

        if prediction == 0:
            down += 1
            if down == 3:
                if initial == -1:
                    initial = 0
                if current_move == 2:
                    counter += 1
                current_move = 0
            elif down > 0:
                up = 0
                no_move = 0
        elif prediction == 2:
            up += 1
            if up == 3 and initial != -1:
                current_move = 2
            elif up > 1:
                down = 0
                no_move = 0
        else:
            no_move += 1
            if no_move == 15:
                current_move = 1
            elif no_move > 10:
                up = 0
                down = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 400)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 5
        # cv2.imshow("wow", frame2)
        # out.write(frame2)
        prvs = next

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # 좌측 지표값
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # 좌측 각도계산
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_waist_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            # 우측 지표값
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # 우측 각도계산
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            right_waist_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            if ex == 1:
                # stage 및 피드백
                if left_elbow_angle < 110:
                    stage = "down"
                    a.append(left_elbow_angle)
                    b.append(left_shoulder_angle)
                    c.append(right_elbow_angle)
                    d.append(right_shoulder_angle)
                if left_elbow_angle > 130 and stage == 'down':
                    stage = "up"
                    tmp = min(a)
                    index = a.index(tmp)
                    if LR.predict([[tmp, b[index], c[index], d[index]]]) == 1:
                        print("잘했습니다")
                    else:
                        print("더 숙이세요")
                    a.clear()
                    b.clear()
                    c.clear()
                    d.clear()
            else:
                # stage 및 피드백
                if left_knee_angle < 110:
                    stage = "down"
                    a.append(left_waist_angle)
                    b.append(left_knee_angle)
                    c.append(right_waist_angle)
                    d.append(right_knee_angle)
                if left_knee_angle > 130 and stage == 'down':
                    stage = "up"
                    tmp = min(b)
                    index = b.index(tmp)
                    if LR2.predict([[a[index], tmp, c[index], d[index]]]) == 1:
                        print("잘했습니다")
                    else:
                        print("허리가 너무 굽었거나 자세를 더 낮춰야 합니다")
                    a.clear()
                    b.clear()
                    c.clear()
                    d.clear()
        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (265, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (100, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (95, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        step='push-up'
        if ex==-1:
            step='squat'
        cv2.putText(image, step,
                    (400, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if counter == 10:
            cv2.putText(image, 'Done!', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Mediapipe Feed', image)
            cv2.waitKey(2)
            time.sleep(2)
            ex = ex * -1
            if ex == 1:
                cv2.putText(image, 'Push-Up Time!', (170, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Squat Time!', (170, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            counter = 0
            flag = False

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mediapipe Feed', 1000, 750)
        cv2.imshow('Mediapipe Feed', image)
        if flag == False:
            if flag2 == False:
                cv2.putText(image, 'First, Push-Up Time!', (150, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Mediapipe Feed', image)
            cv2.waitKey(2)
            time.sleep(2)
            flag = True
            flag2 = True

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()