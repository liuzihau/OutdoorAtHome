

from flask import Flask, render_template, Response
import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from exercise import TypeOfExercise
from sounds.sound import fitness_sound
from interface.interface import TypeOfControl
from game.game import *
import pygame
##子豪新增部分  穿衣
import os
import subprocess
import threading
import math
from virtualtryon.tryon import *
##子豪新增部分  穿衣

app=Flask(__name__)

#ap = argparse.ArgumentParser()
#ap.add_argument("-t",
#                "--exercise_type",
#                type=str,
#                help='Type of activity to do',
#                required=True)
#ap.add_argument("-vs",
#                "--video_source",
#                type=str,
#                help='Type of activity to do',
#                required=False)
#args = vars(ap.parse_args())


## 強制進首頁使用control模式
args = {}
args['video_source'] = None
args["exercise_type"] = 'control'
args['type'] = 'fitness'


## drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def control_frames():  
    ## setting the video source
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(0)  # webcam
    w = 640
    h = 480
    cap.set(3, w)  # width
    cap.set(4, h)  # height

    with mp_pose.Pose(min_detection_confidence=0.8,
                  min_tracking_confidence=0.8) as pose:
        counter = 0  # movement of exercise
        status = True  # state of move
        hint = "Ready!"
        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)
            frame = cv2.flip(frame,1)
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
                counter, status, hint = TypeOfControl(landmarks).calculate_exercise(
                    args["exercise_type"], counter, status, hint)
            except:
                pass

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

            #score_frame = score_table(args["exercise_type"], counter, status, hint)
            #print(frame.shape,score_frame.shape)
            #im_h_resize = cv2.hconcat([frame, score_frame])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def fitness_frames(exercise_type):  
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(1)  # webcam
    w = 960
    h = 720
    cap.set(3, w)  # width
    cap.set(4, h)  # height

    ## 音效初始
    music = fitness_sound()

    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as pose:

        counter = 0  # movement of exercise
        status = True  # state of move
        hint = "Ready!"
        start_time = time.time()

        while cap.isOpened():
                    
            ret, frame = cap.read()
            if not ret:
                print("no ret")
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
                counter, status, hint= TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status, hint)            
                music.play_my_sound(exercise_type,int(timer(start_time)[-2:]))
                
                print(int(timer(start_time)[-2:]))
            except:            
                pass

            score_table(exercise_type, counter, status, hint, timer(start_time))
            
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

            # score_frame = score_table(exercise_type, counter, status, hint)
            # score_frame = cv2.resize(score_frame, (320,720), interpolation=cv2.INTER_AREA)
            # print(frame.shape,score_frame.shape)
            # im_h_resize = cv2.hconcat([frame, score_frame])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()            
    # 文字資訊寫入txt檔
            with open('fitness.txt','w+') as f:
                f.write(f"{exercise_type},{counter},{status},{hint}"+'\n')
    # 網頁生成webcam影像
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def games_frames(game_type):
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(0)  # webcam
    w = 1280
    h = 960
    cap.set(3, w)  # width
    cap.set(4, h)  # height   
    
    #音效初始
    file = f"sounds/game1.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    soundon = 0
    
    #設置遊戲初始環境
    start_time = time.time()
    env_list = game_start(game_type)
    counter = 0 # movement of exercise
    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)
            frame = cv2.flip(frame,1)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            ## recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            ## make detection
            results = pose.process(frame)
            ## recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #================================================================

            #遊戲開始進行之後回傳給系統的參數
            env_coordinate = game_play(game_type,env_list,time.time()-start_time)
            #參數被畫在畫布上的樣子
            frame = game_plot(game_type,frame,env_coordinate)
            #================================================================
            try:
                if soundon==0 :
                    pygame.mixer.music.play()
                    soundon = 1
                    start_time = time.time()

                landmarks = results.pose_landmarks.landmark
                total_status = []
                for i,env in enumerate(env_coordinate):
                    counter, env_list[i].status = TypeOfMove(landmarks).calculate_exercise(game_type, counter, env[0],[w,h,env[1],env[2],env[3],env[4]])
                    total_status.append(env_list[i].status)

            except:
                total_status = []
                pass

            # score_table(game_type, counter, [str(x)[0] for x in total_status],timer(start_time))

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

            ## 虛擬手套繪製
            try:
                if game_type == 'game1':
                    angle = BodyPartAngle(landmarks)           

                    x_LEFT_WRIST = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_WRIST'].value].x)
                    y_LEFT_WRIST = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_WRIST'].value].y)
                    # cv2.putText(frame, str(f'{x_LEFT_WRIST},{y_LEFT_WRIST}'), (x_LEFT_WRIST+40, y_LEFT_WRIST+40),cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
                    x_RIGHT_WRIST = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_WRIST'].value].x)
                    y_RIGHT_WRIST = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_WRIST'].value].y)
                    #cv2.putText(frame, str(f'{x_RIGHT_WRIST},{y_RIGHT_WRIST}'), (x_RIGHT_WRIST+40, y_RIGHT_WRIST+40),cv2.FONT_HERSHEY_PLAIN, 2, (212, 255, 127), 2)

                    x_LEFT_ELBOW = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].x)
                    y_LEFT_ELBOW = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].y)
                    #print(f'LEFT_ELBOW[{x_LEFT_ELBOW},{y_LEFT_ELBOW}]')
                    cv2.putText(frame, str(round(angle.angle_of_the_left_arm())), (x_LEFT_ELBOW-20, y_LEFT_ELBOW-20),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (212, 255, 127), 2)
                    cv2.putText(frame, str(round(angle.left_angle_of_the_elbow_horizon())), (x_LEFT_ELBOW+60, y_LEFT_ELBOW+60),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (54, 38, 227), 2)

                    x_RIGHT_ELBOW = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].x)
                    y_RIGHT_ELBOW = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].y)
                    cv2.putText(frame, str(round(angle.angle_of_the_right_arm())), (x_RIGHT_ELBOW-20, y_RIGHT_ELBOW-20),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (212, 255, 127), 2)
                    cv2.putText(frame, str(round(angle.right_angle_of_the_elbow_horizon())), (x_RIGHT_ELBOW+60, y_RIGHT_ELBOW+20),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (54, 38, 227), 2)

                    x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                    y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                    #print(f'LEFT_INDEX[{x_LEFT_INDEX},{y_LEFT_INDEX}]')
                    x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                    y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                    #print(f'RIGHT_INDEX[{x_RIGHT_INDEX},{y_RIGHT_INDEX}]')    
                    x_RIGHT_KNEE = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].x)
                    y_RIGHT_KNEE = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].y) 
                    x_RIGHT_HIP = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].x)
                    y_RIGHT_HIP = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].y) 
                    distance_of_KNEE_HIP=int(abs(((x_RIGHT_KNEE-x_RIGHT_HIP)**2+(y_RIGHT_KNEE-y_RIGHT_HIP)**2)**0.5))

                    # 左手
                    # 載入手套的圖片
                    if round(angle.left_angle_of_the_elbow_horizon()) <= 105 :                        
                        glove = cv2.imread("images/glove1.png")  
                        glove = cv2.flip(glove,1)          
                        img_height, img_width, _ = glove.shape
                        #手套的參考長度          
                        glove_size= (distance_of_KNEE_HIP*2)
                        print(f'glove_size:{glove_size}')            
                        #圖片轉換成適合的大小
                        glove = cv2.resize( glove, (glove_size, glove_size),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)                
                        # 第一個參數旋轉中心(圖片中心)，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
                        #print(f'y_LEFT_ELBOW {y_LEFT_ELBOW},y_LEFT_WRIST {y_LEFT_WRIST}')
                        if y_LEFT_ELBOW >= y_LEFT_WRIST:                
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), -round(angle.left_angle_of_the_elbow_horizon()-90), 1.0)
                        else:                 
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), round(angle.left_angle_of_the_elbow_horizon()+90), 1.0)
                        # 第三個參數變化後的圖片大小
                        glove = cv2.warpAffine(glove, M, (glove_size, glove_size))            
                        #return rotate_img
                        #print(glove.shape)
                        # 透過一系列的處理將眼睛圖片貼在手上
                        glove_gray = cv2.cvtColor(glove, cv2.COLOR_BGR2GRAY)            
                        _, glove_mask = cv2.threshold(glove_gray, 25, 255, cv2.THRESH_BINARY_INV)
                        x, y = int(x_LEFT_INDEX-glove_size/2), int(y_LEFT_INDEX-glove_size/2)#xy是圖的左上角                
                        glove_area = frame[y: y+glove_size, x: x+glove_size]            
                        glove_area_no_glove = cv2.bitwise_and(glove_area, glove_area, mask=glove_mask)            
                        final_glove = cv2.add(glove_area_no_glove, glove)            
                        frame[y: y+glove_size, x: x+glove_size] = final_glove
                    else:
                        glove = cv2.imread("images/glove2.png")                         
                        img_height, img_width, _ = glove.shape
                        #手套的參考長度          
                        glove_size= (distance_of_KNEE_HIP*2)
                        print(f'glove_size:{glove_size}')            
                        #圖片轉換成適合的大小
                        glove = cv2.resize( glove, (glove_size, glove_size),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)                
                        # 第一個參數旋轉中心(圖片中心)，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
                        #print(f'y_LEFT_ELBOW {y_LEFT_ELBOW},y_LEFT_WRIST {y_LEFT_WRIST}')
                        if y_LEFT_ELBOW >= y_LEFT_WRIST:                
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), -round(angle.left_angle_of_the_elbow_horizon()-90), 1.0)
                        else:                 
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), round(angle.left_angle_of_the_elbow_horizon()+90), 1.0)
                        # 第三個參數變化後的圖片大小
                        glove = cv2.warpAffine(glove, M, (glove_size, glove_size))            
                        #return rotate_img
                        #print(glove.shape)
                        # 透過一系列的處理將眼睛圖片貼在手上
                        glove_gray = cv2.cvtColor(glove, cv2.COLOR_BGR2GRAY)            
                        _, glove_mask = cv2.threshold(glove_gray, 25, 255, cv2.THRESH_BINARY_INV)
                        x, y = int(x_LEFT_INDEX-glove_size/2), int(y_LEFT_INDEX-glove_size/2)#xy是圖的左上角                
                        glove_area = frame[y: y+glove_size, x: x+glove_size]            
                        glove_area_no_glove = cv2.bitwise_and(glove_area, glove_area, mask=glove_mask)            
                        final_glove = cv2.add(glove_area_no_glove, glove)            
                        frame[y: y+glove_size, x: x+glove_size] = final_glove 

                    if round(angle.right_angle_of_the_elbow_horizon()) <= 105 :
                        # 右手            
                        glove = cv2.imread("images/glove1.png")
                        img_height, img_width, _ = glove.shape                        
                        glove_size= (distance_of_KNEE_HIP*2)
                        glove = cv2.resize( glove, (glove_size, glove_size),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
                        print(f'y_RIGHT_ELBOW {y_RIGHT_ELBOW},y_RIGHT_WRIST {y_RIGHT_WRIST}')
                        if y_RIGHT_ELBOW >= y_RIGHT_WRIST:                    
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), round(angle.right_angle_of_the_elbow_horizon()-90), 1.0)
                        else:                 
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), -round(angle.right_angle_of_the_elbow_horizon()+90), 1.0)

                        # M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), round(angle.right_angle_of_the_elbow_horizon()-90), 1.0)
                        glove = cv2.warpAffine(glove, M, (glove_size, glove_size))            
                        glove_gray = cv2.cvtColor(glove, cv2.COLOR_BGR2GRAY)            
                        _, glove_mask = cv2.threshold(glove_gray, 25, 255, cv2.THRESH_BINARY_INV)
                        x, y = int(x_RIGHT_INDEX-glove_size/2), int(y_RIGHT_INDEX-glove_size/2)#xy是圖的左上角                
                        glove_area = frame[y: y+glove_size, x: x+glove_size]            
                        glove_area_no_glove = cv2.bitwise_and(glove_area, glove_area, mask=glove_mask)            
                        final_glove = cv2.add(glove_area_no_glove, glove)            
                        frame[y: y+glove_size, x: x+glove_size] = final_glove
                    else:
                        glove = cv2.imread("images/glove2.png")  
                        glove = cv2.flip(glove,1)                                     
                        img_height, img_width, _ = glove.shape                        
                        glove_size= (distance_of_KNEE_HIP*2)
                        glove = cv2.resize( glove, (glove_size, glove_size),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
                        print(f'y_RIGHT_ELBOW {y_RIGHT_ELBOW},y_RIGHT_WRIST {y_RIGHT_WRIST}')
                        if y_RIGHT_ELBOW >= y_RIGHT_WRIST:                    
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), round(angle.right_angle_of_the_elbow_horizon()-90), 1.0)
                        else:                 
                            M = cv2.getRotationMatrix2D((glove_size // 2, glove_size // 2), -round(angle.right_angle_of_the_elbow_horizon()+90), 1.0)
                        glove = cv2.warpAffine(glove, M, (glove_size, glove_size))            
                        glove_gray = cv2.cvtColor(glove, cv2.COLOR_BGR2GRAY)            
                        _, glove_mask = cv2.threshold(glove_gray, 25, 255, cv2.THRESH_BINARY_INV)
                        x, y = int(x_RIGHT_INDEX-glove_size/2), int(y_RIGHT_INDEX-glove_size/2)#xy是圖的左上角                
                        glove_area = frame[y: y+glove_size, x: x+glove_size]            
                        glove_area_no_glove = cv2.bitwise_and(glove_area, glove_area, mask=glove_mask)            
                        final_glove = cv2.add(glove_area_no_glove, glove)            
                        frame[y: y+glove_size, x: x+glove_size] = final_glove             

            except:
                pass

            # score_frame = score_table(game_type, counter, env[0], [w,h,env[1],env[2],env[3],env[4]])
            # score_frame = cv2.resize(score_frame, (320,720), interpolation=cv2.INTER_AREA)
            # print(frame.shape,score_frame.shape)
            # im_h_resize = cv2.hconcat([frame, score_frame])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()

            # 文字資訊寫入txt檔
            with open('game.txt','w+') as f:
                f.write(f"{game_type},{counter}"+'\n')
            # 生成二進為圖檔    
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

##子豪新增部分  叫出命令列跑副程式
def run_win_cmd(cmd):
    result = []
    process = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    for line in process.stdout:
        result.append(line)
    errcode = process.returncode
    for line in result:
        print(line)
    if errcode is not None:
        raise Exception('cmd %s failed, see above for details', cmd)
    return True

##子豪新增部分  跑穿衣的影像
def tryon_frames():
    ## music setting (if needed)
    file = f"sounds/fashion.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    soundon = 0

    # ## drawing body
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    ## setting the video source
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(0)  # webcam

    w = 1600
    h = 1200
    cap.set(3, w)  # width
    cap.set(4, h)  # height
    #設置初始環境
    start_time = time.time()
    env_list = tryon_start('tryon1')
    counter = 0 # movement of exercise
    tryon_status = 1
    page = 0
    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)
            frame = cv2.flip(frame,1)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            ## recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            ## make detection
            results = pose.process(frame)
            ## recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #================================================================
            if tryon_status == 0:
                start_time = time.time()
                env_list = tryon_start('tryon1')
                counter = 0 # movement of exercise
                tryon_status = 1
            elif tryon_status == 1:
                #試穿進行之後回傳給系統的參數
                env_coordinate = tryon_play('tryon1',env_list,time.time()-start_time)
                #參數被畫在畫布上的樣子
                frame = tryon_plot('tryon1',frame,env_coordinate)
            #================================================================
                try:
                    if soundon==0 :
                        pygame.mixer.music.play()
                        soundon = 1
                        start_time = time.time()

                    landmarks = results.pose_landmarks.landmark
                    total_choice = []
                    total_status = []
                    for i,env in enumerate(env_list):
                        counter, env.status , tryon_status,page = TypeOfTry(landmarks).calculate_exercise(
                        'tryon1', counter,env.status, env,w,h,tryon_status,0)
                        total_status.append(env.status)
                        total_choice.append(env.choice)

                except: 
                    total_status = []
                    total_choice = []
                    pass

                # score_table('tryon1', counter, [str(x)[0] for x in total_status],[str(x)[0] for x in total_choice],timer(start_time))
                tryon2start = time.time()
            elif tryon_status == 2 :
                try:
                    landmarks = results.pose_landmarks.landmark
                    if len(env_list):
                        [frame , tryon2start] = tryon2_plot('tryon1',frame,env_list,w,h,tryon2start)
                    else:
                        tryon_status = 3
                        path="VITON-HD/datasets/play/cloth"
                        [product_list,env_list] = add_product(path)
                        start_time = time.time()
                        page = 0
                        max_page = math.ceil(len(product_list)/4)
                        
                except: 
                    pass
            elif tryon_status == 3:
                work_list = []
                if page > max_page:
                    page = max_page
                elif page < 0:
                    page = 0
                work_list.extend(product_list[page*4:page*4+4])
                work_list.extend(env_list)
                env_coordinate = tryon_play('tryon1',work_list,time.time()-start_time)
                frame = tryon_plot('tryon1',frame,env_coordinate,path)
                try:
                    landmarks = results.pose_landmarks.landmark
                    for i,env in enumerate(work_list):
                        counter, env.status , tryon_status, page = TypeOfTry(landmarks).calculate_exercise(
                        'tryon1', counter,env.status, env,w,h,tryon_status,page)
                except: 
                    pass
            elif tryon_status == 4:
                with open('product.txt','w') as f:
                    for obj in product_list:
                        if obj.choice:
                            filename = obj.position.split('-')[-1]
                            f.writelines(f"{filename}.jpg"+'\n')
                break

                

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


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
           

    t = threading.Thread(target=run_win_cmd,args=(f'python subpro.py',))
    t.start()

    cap = cv2.VideoCapture('videos/wait.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # frame = frame.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if not t.is_alive():
            break


        
    t.join()
    final_path ='VITON-HD/results/play'
    final_file = [f"{final_path}/{file}" for file in os.listdir(final_path)]
    total_pics = len(final_file)
    if math.ceil(total_pics/2)*768<2880:
        total_w = 2880
    else:
        total_w = math.ceil(total_pics/2)*768
    total_h = 2160
    final_frame = np.zeros((total_h,total_w,3),dtype=np.uint8)
    for i,show in enumerate(final_file):
        image = cv2.imread(show)
        if i % 2 == 0:
            final_frame[30:1054,(i//2)*768:(i//2)*768+768] = image
        elif i % 2 ==1:
            final_frame[1106:2130,(i//2)*768:(i//2)*768+768] = image
    display_start = time.time() 
    scroll = 80
    while True:
        now_time = time.time()-display_start
        now_w = int(now_time*scroll)
        right_w = min(total_w,now_w+2880)
        if right_w== total_w:
            now_w = right_w-2880
        show_frame = final_frame[:,now_w:right_w]
        show_frame = cv2.resize(show_frame,(1440,1024),interpolation=cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', show_frame)
        frame = buffer.tobytes()
        # frame = frame.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



## 主選單
@app.route('/')
def index():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('index.html')

## control頁面影像
@app.route('/video_feed')
def video_feed():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return Response(control_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


## 運動頁面
@app.route('/fitness/<string:exercise_type>')
def fitness(exercise_type):
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('fitness.html',exercise_type=exercise_type)

## 遊戲頁面
@app.route('/game/<string:game_type>')
def game(game_type):
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('game.html',game_type=game_type)

## 子豪新增部分 穿衣頁面
@app.route('/tryon_stage')
def tryon_stage():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('tryon_stage.html',title = 'tryon_feed')

## 運動頁面影像
@app.route('/fitness_feed/<string:exercise_type>')
def fitness_feed(exercise_type):
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return Response(fitness_frames(exercise_type), mimetype='multipart/x-mixed-replace; boundary=frame')

## 遊戲頁面影像
@app.route('/games_feed/<string:game_type>')
def games_feed(game_type):
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return Response(games_frames(game_type), mimetype='multipart/x-mixed-replace; boundary=frame')

## 子豪新增部分  穿衣頁面影像
@app.route('/tryon_feed')
def tryon_feed():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return Response(tryon_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

## 健身選單
@app.route('/sport')
def sport():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('sport.html')

## 遊戲選單
@app.route('/game_menu')
def game_menu():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('game_menu.html')


# ## 測試選單
# @app.route('/test')
# def test():
#     return render_template('test.html',title = 'fitness_feed/squat')

## 健身傳字頁面
@app.route('/status_feed')
def status_feed():
    def generate():
        with open('fitness.txt','r') as f:
            yield f.read()  # return also will work
    return Response(generate(), mimetype='text') 

## 遊戲傳字頁面
@app.route('/game_status_feed')
def game_status_feed():
    def generate():
        with open('game.txt','r') as f:
            yield f.read()  # return also will work
    return Response(generate(), mimetype='text') 


if __name__=='__main__':
    import argparse
    app.run(debug=True)
