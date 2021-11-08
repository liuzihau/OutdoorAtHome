from datetime import datetime
from flask import Flask, request, render_template, session, redirect, Response, flash
from datetime import timedelta
import cv2
# import argparse
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
from dance_feed import game3_frames
# from bgadd import *

app=Flask(__name__)

app.config['SECRET_KEY'] = '12345'  # 設定session加密的金鑰

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


## 控制首頁隱藏影像
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


## 健身影像
def fitness_frames(exercise_type):  
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(0)  # webcam
    w = 960
    h = 720
    cap.set(3, w)  # width
    cap.set(4, h)  # height    
    
    mp4=f"videos/{exercise_type}.mp4"    
    cap = cv2.VideoCapture(mp4)
    counter = 0  # movement of exercise
    status = True  # state of move
    hint = "Ready!"
    switch=True
    soundon = 0
    flag = 0 
    # mp3=f"sounds/{exercise_type}.mp3"
    # pygame.mixer.init()
    # pygame.mixer.music.load(mp3)
    
    # while cap.isOpened():
        
    #     #print(exercise_type)
               
    #     if soundon == 0 :
    #         pygame.mixer.music.play()            
    #         soundon = 1
    #     try:
    #         ret, frame = cap.read() 
    #         frame = cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_AREA)
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #             # frame = frame.tobytes()            
    #         # 文字資訊寫入txt檔
                
    #         # 網頁生成webcam影像
    #         cv2.waitKey(1) #<--從25改1,比較順
    #         with open('fitness.txt','w+') as f:
    #             f.write(f"{switch},{exercise_type},{counter},{status},{hint}"+'\n')
    #         yield (b'--frame\r\n'
    #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #     except:
    #         break  

    cap = cv2.VideoCapture(0)
    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as pose:

        encourage=["images/super.png", "images/greatjob.png", "images/goodjob1.png", "images/goodjob.png",
         "images/welldown.png", "images/awesome.png","images/nicework.png" ]
        start_time = time.time()
        while cap.isOpened():
                    
            ret, frame = cap.read()     
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
                #counter+=0.1
                landmarks = results.pose_landmarks.landmark            
                counter, status, hint= TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status, hint)                
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
            if int(counter)%5==0 and flag == 0:
                enc_img = cv2.imread(random.choice(encourage))
                flag = 1    
            if 0<round(counter,1)%5.0<2:
                enc_img = cv2.resize( enc_img, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
                img_height, img_width, _ = enc_img.shape                           
                enc_img_gray = cv2.cvtColor(enc_img, cv2.COLOR_BGR2GRAY)            
                _, enc_img_mask = cv2.threshold(enc_img_gray, 25, 255, cv2.THRESH_BINARY_INV)
                x, y = int(300-img_width/2), int(400-img_height/2)              
                enc_img_area = frame[y: y+img_height, x: x+img_width]            
                enc_img_area_no_enc_img = cv2.bitwise_and(enc_img_area, enc_img_area, mask=enc_img_mask)            
                final_enc_img = cv2.add(enc_img_area_no_enc_img, enc_img)            
                frame[y: y+img_height, x: x+img_width] = final_enc_img
                flag = 0      
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
                       
    # 文字資訊寫入txt檔
            if round(counter,1)>=20:
                switch=False
            #print(switch)
            end_time = time.time()-start_time

            with open('fitness.txt','w+') as f:
                f.write(f"{switch},{exercise_type},{counter},{status},{hint},{end_time}"+'\n')
                #print(end_time)
    # 網頁生成webcam影像
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


## 拳擊遊戲game1影像
def games_frames(game_type='game1'):
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
    game_status = 0 #start644

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

            if timer(start_time) == "00:00:55":
                game_status = 1 #end

            # score_frame = score_table(game_type, counter, env[0], [w,h,env[1],env[2],env[3],env[4]])
            # score_frame = cv2.resize(score_frame, (320,720), interpolation=cv2.INTER_AREA)
            # print(frame.shape,score_frame.shape)
            # im_h_resize = cv2.hconcat([frame, score_frame])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()

            # 文字資訊寫入txt檔
            with open('game.txt','w+') as f:
                f.write(f"{game_status},{game_type},{counter},{timer(start_time)}"+'\n')
            # 生成二進為圖檔    
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


## 一二三我姑媽game2影像
def game2_frames(game_type='game2'):
    def timer1(start_time):
        time_diff = time.time()-start_time
        return time_diff
    file = f"sounds/game2.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    game_over = False
    soundon = 0
    start = 0
    total_use_time = 0
    game_status = 0 # game start
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_pose = mp.solutions.pose

    #設置遊戲初始環境
    start_time = time.time()
    pose = mp_pose.Pose(min_detection_confidence=0.8,
                        min_tracking_confidence=0.8)
    counter = 0 
    hard = 21
    time_list = [8*g+x for g,x in enumerate(sorted(random.sample(range(4,70),20)))]
    print(time_list)

    final_frame = np.ones((800,1320,3),dtype =np.uint8)*60
    print(final_frame.shape)
    cap = cv2.VideoCapture(0)
    status = True

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
        bg_image = None
        while cap.isOpened():
        #=== 作弊程式 ====
            # if counter<hard-1:
                # counter += 0.2
        #===================
            if soundon== 0 :
                if int(timer1(start_time)) in time_list:
                    pygame.mixer.music.play()
                    soundon = 1
                    act_time = time.time()

            print(counter)
            if soundon == 1 and 4.7<(time.time()-act_time)<8:
                L_IMAGE = cv2.imread('images/007.png')
                L_IMAGE = cv2.resize(L_IMAGE,(640,780),interpolation=cv2.INTER_LINEAR)
            elif soundon == 1 and (time.time()-act_time)>8:
                soundon=0
            else:
                L_IMAGE = cv2.imread('images/006.png')
            if game_over:
                game_status = 1 #fail
                soundon = 2
                end_time = time.time()
                total_use_time = timer(start_time,end_time)
                break
                img = cv2.imread('images/008.png')
                h_center = int(img.shape[0]/2)
                w_center = int(img.shape[1]/2)
                dead_h = img.shape[0]
                dead_w = img.shape[1]
                ratio = min(1,(1+dead_time)/8)
                print(ratio)
                img = img[int(h_center-ratio*dead_h/2):int(h_center+ratio*dead_h/2),int(w_center-ratio*dead_w/2):int(w_center+ratio*dead_w/2)]
                final_frame = cv2.resize(img,(1320,800),interpolation=cv2.INTER_LINEAR)
    
            elif (hard-1-0.01)<counter<(hard-1+0.22):
                game_status = 2 #win
                soundon = 2

                break
                if start == 0:
                    cap = cv2.VideoCapture('images/victory.mp4')
                    start +=1
                elif cap.get(cv2.CAP_PROP_POS_FRAMES) <227:
                    ret , final_frame = cap.read()
                    final_frame = cv2.resize(img,(1320,800),interpolation=cv2.INTER_LINEAR)
                    end_time = time.time()
                else:
                    vic_time = time.time()-end_time
                    img = cv2.imread('images/victory.png')
                    total_use_time = timer(start_time,end_time)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,f"Total time : {total_use_time}",(320,600),font,2,(80,127,255),3,cv2.LINE_AA)
                    h_center = int(img.shape[0]/2)
                    w_center = int(img.shape[1]/2)
                    vic_h = img.shape[0]
                    vic_w = img.shape[1]
                    ratio = min(1,(1+vic_time)/8)
                    img = img[int(h_center-ratio*vic_h/2):int(h_center+ratio*vic_h/2),int(w_center-ratio*vic_w/2):int(w_center+ratio*vic_w/2)]
                    final_frame = cv2.resize(img,(1320,800),interpolation=cv2.INTER_LINEAR)

            else:
                BG_COLOR = cv2.imread("images/004.png")
                RL_IMAGE = cv2.imread('images/005.png')
                RL_x = int(0+(700/hard)*counter)
                RL_y = int(0+(900/hard)*counter)
                RL_Rangex = int(2560-1920*counter/hard)
                RL_Rangey = int(1200-900*counter/hard)
                RL_IMAGE = RL_IMAGE[RL_y:RL_y+RL_Rangey,RL_x:RL_x+RL_Rangex]
                RL_IMAGE = cv2.resize(RL_IMAGE,(640,300), interpolation=cv2.INTER_AREA)
                Orix = int(1533-(893/hard)*counter)
                Oriy = int(1150-(670/hard)*counter)
                BG_COLOR = cv2.resize(BG_COLOR, (Orix, Oriy), interpolation=cv2.INTER_AREA)
                x = int(447-(447/hard)*counter)
                y = int(335-(335/hard)*counter)
                w = 640
                h = 480
                BG_COLOR = BG_COLOR[y:y+h, x:x+w]



                # print(counter)
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.resize(image,(640,480))
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = selfie_segmentation.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw selfie segmentation on the background image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                condition = np.stack(
                  (results.segmentation_mask,) * 3, axis=-1) > 0.1
                # The background can be customized.
                #   a) Load an image (with the same width and height of the input image) to
                #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
                #   b) Blur the input image by applying image filtering, e.g.,
                #      bg_image = cv2.GaussianBlur(image,(55,55),0)
                # if bg_image is None:
                #   bg_image = np.zeros(image.shape, dtype=np.uint8)
                #   bg_image[:] = BG_COLOR
                output_image = np.where(condition, image, BG_COLOR)
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                output_image.flags.writeable = False
                results = pose.process(output_image)
                output_image.flags.writeable = True
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)


                try:
                    landmarks = results.pose_landmarks.landmark
                    counter, status= TypeOfMove(landmarks).calculate_exercise(game_type, counter, status)
                    # print('landmark')
                    if 4.4<time.time()-act_time<5.1:
                        [LA,RA,LL,RL,N,AB] = BodyPartAngle(landmarks).angle_of_the_left_arm(),BodyPartAngle(landmarks).angle_of_the_right_arm(),BodyPartAngle(landmarks).angle_of_the_left_leg(),BodyPartAngle(landmarks).angle_of_the_right_leg(),BodyPartAngle(landmarks).angle_of_the_neck(),BodyPartAngle(landmarks).angle_of_the_abdomen()
                        [LWRV,RWRV,LELV,RELV,LKNV,RKNV] = detection_body_part(landmarks, "LEFT_WRIST")[2],detection_body_part(landmarks, "RIGHT_WRIST")[2],detection_body_part(landmarks, "LEFT_ELBOW")[2],detection_body_part(landmarks, "RIGHT_ELBOW")[2],detection_body_part(landmarks, "LEFT_KNEE")[2],detection_body_part(landmarks, "RIGHT_KNEE")[2]
                        # print([LA,RA,LL,RL,N,AB])
                        # print([LWRV,RWRV,LELV,RELV,LANV,RANV])
                    if 4.6<time.time()-act_time<8:
                        test = max(abs(LA-BodyPartAngle(landmarks).angle_of_the_left_arm()),abs(RA-BodyPartAngle(landmarks).angle_of_the_right_arm()),abs(LL-BodyPartAngle(landmarks).angle_of_the_left_leg()),abs(RL-BodyPartAngle(landmarks).angle_of_the_right_leg()),abs(N-BodyPartAngle(landmarks).angle_of_the_neck()),abs(AB-BodyPartAngle(landmarks).angle_of_the_abdomen()))
                        Vtest = np.mean([LWRV,RWRV,LELV,RELV,LKNV,RKNV])
                        print(Vtest)
                        # print(test)
                        if test > 10 and Vtest>0.7 :
                            game_over = True
                            over_time = time.time()

                    # print('final')
                except:
                    pass

                mp_drawing.draw_landmarks(
                        output_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                thickness=2,
                                                circle_radius=2),
                        mp_drawing.DrawingSpec(color=(174, 139, 45),
                                                thickness=2,
                                                circle_radius=2),
                    )
                final_frame[5:305,670:1310] = RL_IMAGE
                final_frame[315:795,670:1310] = output_image
                final_frame[10:790,10:650] = L_IMAGE

                ret, buffer = cv2.imencode('.jpg', final_frame)
                frame = buffer.tobytes()

            # game_status: 0=START, 1=LOSE, 2=WIN 
            with open('game.txt','w+') as f:
                f.write(f"{game_status},{game_type},{int(round(counter))},{timer(start_time)},{total_use_time}"+'\n')

            # 網頁生成webcam影像(勝利與失敗畫面會當機，另寫在下面)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

        cap = cv2.VideoCapture('images/victory.mp4')
        while cap.isOpened():
            if cap.get(cv2.CAP_PROP_POS_FRAMES) <227 and game_status == 2:
                ret , final_frame = cap.read()
                final_frame = cv2.resize(final_frame,(1320,800),interpolation=cv2.INTER_LINEAR)
                end_time = time.time()
            elif game_status == 2:
                vic_time = time.time()-end_time
                img = cv2.imread('images/victory.png')
                total_use_time = timer(start_time,end_time)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,f"Total time : {total_use_time}",(320,600),font,2,(80,127,255),3,cv2.LINE_AA)
                h_center = int(img.shape[0]/2)
                w_center = int(img.shape[1]/2)
                vic_h = img.shape[0]
                vic_w = img.shape[1]
                ratio = min(1,(1+vic_time)/2) #原本/8
                img = img[int(h_center-ratio*vic_h/2):int(h_center+ratio*vic_h/2),int(w_center-ratio*vic_w/2):int(w_center+ratio*vic_w/2)]
                final_frame = cv2.resize(img,(1320,800),interpolation=cv2.INTER_LINEAR)
                if ratio ==1:
                    game_status = 3
            else:
                dead_time = time.time()-over_time
                img = cv2.imread('images/008.png')
                h_center = int(img.shape[0]/2)
                w_center = int(img.shape[1]/2)
                dead_h = img.shape[0]
                dead_w = img.shape[1]
                ratio = min(1,(1+dead_time)/2) #原本/8
                print(ratio)
                img = img[int(h_center-ratio*dead_h/2):int(h_center+ratio*dead_h/2),int(w_center-ratio*dead_w/2):int(w_center+ratio*dead_w/2)]
                final_frame = cv2.resize(img,(1320,800),interpolation=cv2.INTER_LINEAR)            
            

                
            ret, buffer = cv2.imencode('.jpg', final_frame)
            frame = buffer.tobytes()

            # game_status: 0=START, 1=LOSE, 2=WIN 
            with open('game.txt','w+') as f:
                f.write(f"{game_status},{game_type},{int(round(counter))},{timer(start_time)},{total_use_time}"+'\n')


            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

## 跳舞遊戲game3影像(太多code，另外引入)

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
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # webcam

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
                with open('product.txt','w+') as f:
                    for obj in product_list:
                        if obj.choice:
                            filename = obj.position.split('-')[-1]
                            f.writelines(f"{filename}.jpg"+'\n')
                break

                

            ## render detections (for landmarks)
            # mp_drawing.draw_landmarks(
            #     frame,
            #     results.pose_landmarks,
            #     mp_pose.POSE_CONNECTIONS,
            #     mp_drawing.DrawingSpec(color=(255, 255, 255),
            #                         thickness=2,
            #                         circle_radius=2),
            #     mp_drawing.DrawingSpec(color=(174, 139, 45),
            #                         thickness=2,
            #                         circle_radius=2),
            # )


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()
            with open('tryon.txt','w+') as f:
                f.write(f"True")
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
    cycle_test = total_pics // 8  #測試會有幾個拼貼流程
    remain_test = total_pics % 8 #測試不成套的圖有幾個
    if cycle_test<1:
        total_w = 2600
    else:
        if remain_test==0:
            # 加尾數32
            total_w = cycle_test*2202+32
        elif remain_test < 4:
            # 加起始32跟前三個拼貼的寬度768再加尾數32
            total_w = cycle_test*2202+32+768+32
        elif remain_test < 6:
            # 追加中間2個拼貼的寬度768再加尾數32
            total_w = cycle_test*2202+32+768+32+570+32
        else:
            # 完整兩塊拼貼版+尾數32
            total_w = cycle_test*2202+2202+32
    total_h = 1620
    final_frame = np.zeros((total_h,total_w,3),dtype=np.uint8)*30
    for i,show in enumerate(final_file):
        i_cycle_test = (i+1)//8
        i_remain_test = (i+1)%8
        image = cv2.imread(show)
        if i_remain_test % 8 == 1:
            final_frame[30:1054,(i_cycle_test*2202+32):(32+i_cycle_test*2202)+768] = image
        elif i_remain_test % 8 == 2:
            image = cv2.resize(image,(364,486),interpolation=cv2.INTER_AREA)
            final_frame[(30+1024+50):(30+1024+50)+486,(i_cycle_test*2202+32):(32+i_cycle_test*2202)+364] = image
        elif i_remain_test % 8 == 3:
            image = cv2.resize(image,(364,486),interpolation=cv2.INTER_AREA)
            final_frame[(30+1024+50):(30+1024+50)+486,(i_cycle_test*2202+32+364+40):(32+i_cycle_test*2202+364+40)+364] = image
        elif i_remain_test % 8 == 4:
            image = cv2.resize(image,(570,760),interpolation=cv2.INTER_AREA)
            final_frame[30:30+760,(i_cycle_test*2202+32+768+32):(i_cycle_test*2202+32+768+32)+570] = image
        elif i_remain_test % 8 == 5:
            image = cv2.resize(image,(570,760),interpolation=cv2.INTER_AREA)
            final_frame[(30+760+40):(30+760+40)+760,(i_cycle_test*2202+32+768+32):(i_cycle_test*2202+32+768+32)+570] = image
        elif i_remain_test % 8 == 6 :
            final_frame[(30+486+50):(30+486+50)+1024,(i_cycle_test*2202+32+768+32+570+32):(i_cycle_test*2202+32+768+32+570+32)+768] = image
        elif i_remain_test % 8 == 7 :
            image = cv2.resize(image,(364,486),interpolation=cv2.INTER_AREA)
            final_frame[30:(30+486),(i_cycle_test*2202+32+768+32+570+32):(i_cycle_test*2202+32+768+32+570+32)+364] = image
        elif i_remain_test % 8 == 0 :
            image = cv2.resize(image,(364,486),interpolation=cv2.INTER_AREA)
            final_frame[30:(30+486),((i_cycle_test-1)*2202+32+768+32+570+32+364+32):((i_cycle_test-1)*2202+32+768+32+570+32+364+32)+364] = image
        fn = show.split('/')[-1]
        os.rename(show,f'history_output\{fn}')
    display_start = time.time() 
    scroll = 80
    end_flag = 0
    end_time = display_start+500
    outputname = round(time.time())
    cv2.imwrite(f'history_output\output{outputname}.jpg',final_frame)
    while True:
        now_time = time.time()-display_start
        now_w = int(now_time*scroll)
        right_w = min(total_w,now_w+2600)
        if right_w== total_w:
            now_w = right_w-2600
            if end_flag == 0 :
                end_time = time.time()
                end_flag = 1
        show_frame = final_frame[:,now_w:right_w]
        show_frame = cv2.resize(show_frame,(1600,1024),interpolation=cv2.INTER_AREA)
        ret, buffer = cv2.imencode('.jpg', show_frame)
        frame = buffer.tobytes()
        # frame = frame.tobytes()
        with open('tryon.txt','w+') as f:
            if time.time()-end_time>5:
                test_product = []
                for obj in product_list:
                    if obj.choice:
                        filename = obj.position.split('-')[-1]
                        test_product.append(f"{filename}.jpg")
                pro_str = ','.join(test_product)
                f.write(f"False,{pro_str}")
            else:
                f.write(f"True")

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



## ====flask路由====
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
## 登入畫面
@app.route('/login', methods=['GET', 'POST'])  # 支援get、post請求
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']   #.encode('utf-8')
        if email == "abc@gmail.com":
            user = "user"
            if user == None :
                # return "沒有這個帳號"
                flash('沒有這個帳號')
                return render_template("login.html")
            if len(user) != 0:
                if password == '12345':
                    session['name'] = 'abc'  
                    session['email'] = 'abc@gmail.com'
                    return render_template("index_3D.html")
                else:
                    # return "您的密碼錯誤"
                    flash('您的密碼錯誤')
                    return render_template("login.html")
        # 以下暫時寫的
        else:
            flash('沒有這個帳號')
            return render_template("login.html")
    else:
        return render_template("login.html")


## 主選單
@app.route('/')
def index():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    # session['name'] = False
    username = session.get('name')  # 取session

    if username == "abc":
        return render_template('index_3D.html')
    else:
        return redirect('/login')

    #return render_template('index.html')


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
    #清空文件檔
    with open('fitness.txt','w+') as f:
        f.write(f""+'\n')

    return render_template('fitness.html',exercise_type=exercise_type)

## 遊戲頁面
@app.route('/game/<string:game_type>')
def game(game_type):
    #清空文件檔
    with open('game.txt','w+') as f:
        f.write(f""+'\n')

    return render_template('game.html',game_type=game_type)

## 子豪新增部分 穿衣頁面
@app.route('/tryon_stage')
def tryon_stage():
    #清空文件檔
    with open('tryon.txt','w+') as f:
        f.write(f"True,"+'\n')
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
    if game_type=='game1':
        # 先關聲音
        pygame.mixer.init()
        pygame.mixer.music.stop()
        return Response(games_frames(game_type), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif game_type=='game2':
        # 先關聲音
        pygame.mixer.init()
        pygame.mixer.music.stop()
        return Response(game2_frames(game_type), mimetype='multipart/x-mixed-replace; boundary=frame')

    elif game_type=="game3":
        pygame.mixer.init()
        pygame.mixer.music.stop()
        return Response(game3_frames(game_type), mimetype='multipart/x-mixed-replace; boundary=frame')


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
    return render_template('sport_3D.html')

## 遊戲選單
@app.route('/game_menu')
def game_menu():
    # 先關聲音
    pygame.mixer.init()
    pygame.mixer.music.stop()
    return render_template('game_menu_3D.html')


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

## 穿衣傳字頁面
@app.route('/tryon_status_feed')
def tryon_status_feed():
    def generate():
        with open('tryon.txt','r') as f:
            yield f.read()  # return also will work
    return Response(generate(), mimetype='text') 

## 登出
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


if __name__=='__main__':
    import argparse
    app.run(debug=True)
