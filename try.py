'''
This file is originally from "Sports With AI" https://github.com/Furkan-Gulsen/Sport-With-AI/blob/main/main.py
'''
import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from fitness.exercise import TypeOfExercise

## setup agrparse
ap = argparse.ArgumentParser()
ap.add_argument("-t",
                "--exercise_type",
                type=str,
                help='Type of activity to do',
                required=True)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Type of activity to do',
                required=False)
args = vars(ap.parse_args())

## drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## setting the video source
if args["video_source"] is not None:
    cap = cv2.VideoCapture(args["video_source"])
else:
    cap = cv2.VideoCapture(0)  # webcam
w = 800
h = 480
cap.set(3, w)  # width
cap.set(4, h)  # height

## setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.8,
                  min_tracking_confidence=0.8) as pose:

    counter = 0  # movement of exercise
    status = True  # state of move
    while cap.isOpened():
        ret, frame = cap.read()
        # result_screen = np.zeros((250, 400, 3), np.uint8)

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ## recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        ## make detection
        results = pose.process(frame)
        ## recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                args["exercise_type"], counter, status)
        except:
            pass

        score_table(args["exercise_type"], counter, status)

        ## render detections (for landmarks)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=2),
            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                   thickness=2,
                                   circle_radius=2),
        )
        try:
            angle = BodyPartAngle(landmarks)
            cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].x)
            cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].y)
            cv2.putText(frame, str(round(angle.angle_of_the_left_arm())), (cx-20, cy-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
            cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].x)
            cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].y)
            cv2.putText(frame, str(round(angle.angle_of_the_right_arm())), (cx-20, cy-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
            cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_KNEE'].value].x)
            cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_KNEE'].value].y)
            cv2.putText(frame, str(round(angle.angle_of_the_left_leg())), (cx-20, cy-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (235, 150, 150), 2)
            cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].x)
            cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].y)
            cv2.putText(frame, str(round(angle.angle_of_the_right_leg())), (cx-20, cy-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (235, 150, 150), 2)
            cx = int(w *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_SHOULDER'].value].x+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_SHOULDER'].value].x)/2)
            cy = int(h *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_SHOULDER'].value].y+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_SHOULDER'].value].y)/2)
            cv2.putText(frame, str(round(angle.angle_of_the_neck())), (cx-20, cy-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (150, 235, 150), 2)
            cx = int(w *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_HIP'].value].x+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].x)/2)
            cy = int(h *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_HIP'].value].y+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].y)/2)
            cv2.putText(frame, str(round(angle.angle_of_the_abdomen())), (cx-20, cy-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150), 2)
        except:
            pass


        # BodyPartAngle.angle_of_the_neck
        # BodyPartAngle.angle_of_the_abdomen


        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
