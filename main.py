from  ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
from angle import calculate_angle
from collections import deque
from log_data import log

neck_his = deque(maxlen=60) #1sec = 30fps
spine_his = deque(maxlen=60)

model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
frame_count = 4
results = None

while True:
    
    ret, frame = cap.read()
    frame_count+=1

    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    if frame_count % 5 == 0:
        results = model(frame, conf=0.5)[0]

    dev_detected = False

    if results is None:
        results = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["cell phone", "laptop"]:
            dev_detected = True

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    results_pose = pose.process(rgb)

    if results_pose.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results_pose.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        h, w, _ = frame.shape

        shoulder = [left_shoulder.x * w, left_shoulder.y * h]
        ear = [left_ear.x * w, left_ear.y * h]
        hip = [left_hip.x * w, left_hip.y * h]

        vertical_ref = [shoulder[0], shoulder[1]-100]

        neck_angle = calculate_angle(ear, shoulder, vertical_ref)
        neck_his.append(neck_angle)
        spine_angle = calculate_angle(hip, shoulder, vertical_ref)
        spine_his.append(spine_angle)

        avg_neck = sum(neck_his)/len(neck_his)
        avg_spine = sum(spine_his)/len(spine_his)

        neck_var = np.var(neck_his)
        spine_var = np.var(spine_his)

        risk = 0

        bad_spine = avg_spine > 15
        bad_neck = avg_neck > 20
        static_posture = spine_var < 8 and neck_var < 8

        if bad_neck or bad_spine:
            if static_posture:
                risk+=60
            else:
                risk+=30


        log("log.csv", neck_angle, spine_angle, avg_neck, avg_spine, dev_detected, risk)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


        cv2.putText(frame, f"Spine: {int(avg_spine)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Neck: {int(avg_neck)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Risk: {risk}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if risk > 50 else (0,255,0), 2)

        if dev_detected:
            cv2.putText(frame, "Device detected", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Dekho", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()