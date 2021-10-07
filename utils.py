'''
This file is originally from "Sports With AI" https://github.com/Furkan-Gulsen/Sport-With-AI/blob/main/utils.py
'''
import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
import time

mp_pose = mp.solutions.pose


# returns an angle value as a result of the given points
# 這個函式給定他a b c 三個點 他會由a連線到b再連線到c 然後返回角bac的角度

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    #先y再x 第一項arctan2求cb直線與x軸夾角  第二項arctan2求ab直線與x軸夾角  兩項相減即為角b
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -\
              np.arctan2(a[1] - b[1], a[0] - b[0])  

    #求出徑度 轉成我們比較習慣的角度
    angle = np.abs(radians * 180.0 / np.pi)

    # check cord sys area
    #永遠只計算夾角較小的一邊
    if angle > 180.0:
        angle = 360 - angle

    return angle


# return body part x,y value
#傳進現在的landmarks資料 還有感興趣的landmark資料的名字
def detection_body_part(landmarks, body_part_name):
    #返回該landmark的x y 還有可見度
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]


# return body_part, x, y as dataframe
#傳進現在的landmarks資料 用pd的DataFrame格式回傳所有landmarks的name x y
def detection_body_parts(landmarks):
    body_parts = pd.DataFrame(columns=["body_part", "x", "y"])

    for i, lndmrk in enumerate(mp_pose.PoseLandmark):
        lndmrk = str(lndmrk).split(".")[1]
        cord = detection_body_part(landmarks, lndmrk)
        body_parts.loc[i] = lndmrk, cord[0], cord[1]

    return body_parts


def score_table(exercise, counter, status,time='00:00:00'):
    #先讀一張圖當作背景
    score_table = cv2.imread("./images/score_table.png")
    #第一行請cv2在這張圖上寫上運動名字
    cv2.putText(score_table, "Activity : " + exercise.replace("-", " "),
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2,
                cv2.LINE_AA)
    #第二行請cv2在這張圖上寫上運動次數
    cv2.putText(score_table, "Counter : " + str(int(counter)), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2, cv2.LINE_AA)
    #第三行請cv2在這張圖上寫上目前的狀態
    cv2.putText(score_table, "Status : " + str(status), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2, cv2.LINE_AA)
    cv2.putText(score_table, "Time : " + str(time), (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (182, 158, 128), 2, cv2.LINE_AA)
    cv2.imshow("Score Table", score_table)

def timer(start_time):
    time_diff = time.time()-start_time
    HR = str(int(time_diff // 3600 // 10 )) + str(int(time_diff // 3600 % 10))
    MIN = str(int(time_diff % 3600 // 60 // 10 )) + str(int(time_diff % 3600 // 60 % 10))
    SEC = str(int(time_diff % 60 // 10 )) +str(int(time_diff % 10))
    return f'{HR}:{MIN}:{SEC}'