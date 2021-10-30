'''
This code refer to http://www.circuspi.com/index.php/2021/06/23/ai-mediapipe-unit3/
'''
import cv2
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from exercise import TypeOfExercise
import time
import pygame


## drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## setting the video source
cap = cv2.VideoCapture(0)  # webcam
w = 1600
h = 960
cap.set(3, w)  # width
cap.set(4, h)  # height

## setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.8,
                  min_tracking_confidence=0.8) as pose:

    while cap.isOpened():
                
        ret, frame = cap.read()
        if not ret:
            print("no ret")
        # result_screen = np.zeros((250, 400, 3), np.uint8)

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)#INTER_CUBIC #INTER_AREA #INTER_LINEAR
        frame = cv2.flip(frame,1)
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


        # BodyPartAngle.angle_of_the_neck
        # BodyPartAngle.angle_of_the_abdomen


        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
