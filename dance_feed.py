## import packages
import cv2
import argparse
import time
import mediapipe as mp
import pygame
from body_part_angle import BodyPartAngle
import flask

def game3_frames(game_type='game3'):
    game_status=0

    def mstimer(start_time):
        time_diff = time.time()-start_time
        HR = int(str(int(time_diff // 3600 // 10 )) + str(int(time_diff // 3600 % 10)))
        MIN = int(str(int(time_diff % 3600 // 60 // 10 )) + str(int(time_diff % 3600 // 60 % 10)))
        SEC = int(str(int(time_diff % 60 // 10 )) +str(int(time_diff % 10)))
        MS = int(((str(time_diff)).split(".")[1])[:3])
        #print( f'{HR}:{MIN}:{SEC}')
        return MIN*60+SEC+(MS*0.001)
    def timer(start_time):
        time_diff = time.time()-start_time
        HR = str(int(time_diff // 3600 // 10 )) + str(int(time_diff // 3600 % 10))
        MIN = str(int(time_diff % 3600 // 60 // 10 )) + str(int(time_diff % 3600 // 60 % 10))
        SEC = str(int(time_diff % 60 // 10 )) +str(int(time_diff % 10))
        #print( f'{HR}:{MIN}:{SEC}')
        return f'{HR}:{MIN}:{SEC}'
    #動作位置
    #拍手&say_no
    x_clap_pos=800
    y_clap_pos=420
    x_say_no_pos=800
    y_say_no_pos=420
    #手碰腳
    x_lh_and_ra_pos=900
    y_lh_and_ra_pos=690
    x_rh_and_la_pos=700
    y_rh_and_la_pos=690
    #舉雙手
    x_raise_hands_l_pos=900
    y_raise_hands_l_pos=150
    x_raise_hands_r_pos=700
    y_raise_hands_r_pos=150
    #left_wave
    x_left_wave1=800
    y_left_wave1=420
    x_left_wave2=810
    y_left_wave2=610
    x_left_wave3=900
    y_left_wave3=620
    x_left_wave4=1070
    y_left_wave4=590
    x_left_wave5=1110
    y_left_wave5=400
    x_left_wave6=1100
    y_left_wave6=220

    #right_wave
    x_right_wave1=800
    y_right_wave1=420
    x_right_wave2=790
    y_right_wave2=610
    x_right_wave3=700
    y_right_wave3=620
    x_right_wave4=530
    y_right_wave4=590
    x_right_wave5=490
    y_right_wave5=400
    x_right_wave6=500
    y_right_wave6=220

    class TypeOfDance(BodyPartAngle):
        def __init__(self, landmarks):
            super().__init__(landmarks)
        #算分數
        def hand_clap(self, counter, status, move=""):

            if move =='clap':
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                distance_of_l_INDEX_r_INDEX = int(abs(((x_LEFT_INDEX-x_RIGHT_INDEX)**2+(y_LEFT_INDEX-y_RIGHT_INDEX)**2)**0.5))
                x_center_of_l_INDEX_r_INDEX=(x_LEFT_INDEX+x_RIGHT_INDEX)/2
                y_center_of_l_INDEX_r_INDEX=(y_LEFT_INDEX+y_RIGHT_INDEX)/2
                distance_of_lr_INDEX_clap_pos=int(abs(((x_clap_pos-x_center_of_l_INDEX_r_INDEX)**2+(y_clap_pos-y_center_of_l_INDEX_r_INDEX)**2)**0.5))
                if distance_of_l_INDEX_r_INDEX < 100 and distance_of_lr_INDEX_clap_pos < 100 :
                    print("clap!!!")             
                    counter += 1
                    status = True       
            elif move == 'raise_hands_left': 
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                dist_of_l_INDEX_r_INDEX = int(abs(((x_LEFT_INDEX-x_RIGHT_INDEX)**2+(y_LEFT_INDEX-y_RIGHT_INDEX)**2)**0.5))
                x_center_of_l_INDEX_r_INDEX=(x_LEFT_INDEX+x_RIGHT_INDEX)/2
                y_center_of_l_INDEX_r_INDEX=(y_LEFT_INDEX+y_RIGHT_INDEX)/2
                dist_of_lr_INDEX_raise_hands_l_pos=int(abs(((x_raise_hands_l_pos-x_center_of_l_INDEX_r_INDEX)**2+(y_raise_hands_l_pos-y_center_of_l_INDEX_r_INDEX)**2)**0.5))
                if dist_of_l_INDEX_r_INDEX < 100 and dist_of_lr_INDEX_raise_hands_l_pos < 200 :             
                    print("LEFT!!!+++++")
                    counter += 1
                    status = True
            elif move == 'raise_hands_right':
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                dist_of_l_INDEX_r_INDEX = int(abs(((x_LEFT_INDEX-x_RIGHT_INDEX)**2+(y_LEFT_INDEX-y_RIGHT_INDEX)**2)**0.5))
                x_center_of_l_INDEX_r_INDEX=(x_LEFT_INDEX+x_RIGHT_INDEX)/2
                y_center_of_l_INDEX_r_INDEX=(y_LEFT_INDEX+y_RIGHT_INDEX)/2
                dist_of_lr_INDEX_raise_hands_r_pos=int(abs(((x_raise_hands_r_pos-x_center_of_l_INDEX_r_INDEX)**2+(y_raise_hands_r_pos-y_center_of_l_INDEX_r_INDEX)**2)**0.5))            
                if dist_of_l_INDEX_r_INDEX < 100 and dist_of_lr_INDEX_raise_hands_r_pos < 200 : 
                    print("RIGHT!!!+++++")            
                    counter += 1
                    status = True
            elif move =='lh_and_ra':            
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_ANKLE = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ANKLE'].value].x)
                y_RIGHT_ANKLE = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ANKLE'].value].y)
                dist_of_l_INDEX_r_ANKLE = int(abs(((x_LEFT_INDEX-x_RIGHT_ANKLE)**2+(y_LEFT_INDEX-y_RIGHT_ANKLE)**2)**0.5))
                x_center_of_l_INDEX_r_ANKLE=(x_LEFT_INDEX+x_RIGHT_ANKLE)/2
                y_center_of_l_INDEX_r_ANKLE=(y_LEFT_INDEX+y_RIGHT_ANKLE)/2
                dist_of_lh_and_ra_pos=int(abs(((x_lh_and_ra_pos-x_center_of_l_INDEX_r_ANKLE)**2+(y_lh_and_ra_pos-y_center_of_l_INDEX_r_ANKLE)**2)**0.5))
                if dist_of_l_INDEX_r_ANKLE < 100 and dist_of_lh_and_ra_pos < 200:             
                    counter += 1
                    status = True
            elif move =='rh_and_la':            
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                x_LEFT_ANKLE = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ANKLE'].value].x)
                y_LEFT_ANKLE = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ANKLE'].value].y)
                dist_of_r_INDEX_l_ANKLE = int(abs(((x_RIGHT_INDEX-x_LEFT_ANKLE)**2+(y_RIGHT_INDEX-y_LEFT_ANKLE)**2)**0.5))
                x_center_of_r_INDEX_l_ANKLE=(x_RIGHT_INDEX+x_LEFT_ANKLE)/2
                y_center_of_r_INDEX_l_ANKLE=(y_RIGHT_INDEX+y_LEFT_ANKLE)/2
                dist_of_rh_and_la_pos=int(abs(((x_rh_and_la_pos-x_center_of_r_INDEX_l_ANKLE)**2+(y_rh_and_la_pos-y_center_of_r_INDEX_l_ANKLE)**2)**0.5))
                if dist_of_r_INDEX_l_ANKLE < 100 and dist_of_rh_and_la_pos < 200:                          
                    counter += 1
                    status = True        
            elif move =='say_no' or 'say_no1':            
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                dist_of_l_INDEX_say_no_pos=int(abs(((x_say_no_pos-x_LEFT_INDEX)**2+(y_say_no_pos-y_LEFT_INDEX)**2)**0.5))
                dist_of_r_INDEX_say_no_pos=int(abs(((x_say_no_pos-x_RIGHT_INDEX)**2+(y_say_no_pos-y_RIGHT_INDEX)**2)**0.5))
                if dist_of_l_INDEX_say_no_pos < 100 or dist_of_r_INDEX_say_no_pos < 100 :             
                    counter += 1
                    status = True
            elif move =='say_no' or 'say_no1':            
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                dist_of_l_INDEX_say_no_pos=int(abs(((x_say_no_pos-x_LEFT_INDEX)**2+(y_say_no_pos-y_LEFT_INDEX)**2)**0.5))
                dist_of_r_INDEX_say_no_pos=int(abs(((x_say_no_pos-x_RIGHT_INDEX)**2+(y_say_no_pos-y_RIGHT_INDEX)**2)**0.5))
                if dist_of_l_INDEX_say_no_pos < 100 or dist_of_r_INDEX_say_no_pos < 100 :             
                    counter += 1
                    status = True

            return [counter, status, move]

    #定義show圖位置
    def Move(frame,move):

        if move == 'clap':
            clap = cv2.imread("images/clap111.png")
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_clap_pos-img_width/2), int(y_clap_pos-img_height/2)#xy是圖的左上角 #800*330               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("clap")
            return frame

        elif move == 'lh_and_ra':
            clap = cv2.imread("images/handankle.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_lh_and_ra_pos-img_width/2), int(y_lh_and_ra_pos-img_height/2) #900*660               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print('lh_and_ra')
            return frame
        elif move == 'rh_and_la':
            clap = cv2.imread("images/handankle.png")  
            clap = cv2.flip(clap,1)      
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_rh_and_la_pos-img_width/2), int(y_rh_and_la_pos-img_height/2)#700*660                
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("rh_and_la")
            return frame
        elif move == 'say_no':
            clap = cv2.imread("images/say_no.png")
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_say_no_pos-img_width/2), int(y_say_no_pos-img_height/2)#xy是圖的左上角 #800*330               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("say_no")
            return frame
        elif move == 'say_no1':
            clap = cv2.imread("images/say_no1.png")
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_say_no_pos-img_width/2), int(y_say_no_pos-img_height/2)#xy是圖的左上角 #800*330               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("say_no1")
            return frame    
        elif move == 'raise_hands_left':
            clap = cv2.imread("images/raise_hands.png")
            clap = cv2.flip(clap,1)
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_raise_hands_l_pos-img_width/2), int(y_raise_hands_l_pos-img_height/2)#xy是圖的左上角 #900*125               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("raise_hands_left")
            return frame
        elif move == 'raise_hands_right':
            clap = cv2.imread("images/raise_hands.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_raise_hands_r_pos-img_width/2), int(y_raise_hands_r_pos-img_height/2)#xy是圖的左上角 #700*125               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("raise_hands_right")
            return frame
        elif move == 'left_wave1':
            clap = cv2.imread("images/wave1.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_left_wave1-img_width/2), int(y_left_wave1-img_height/2)#xy是圖的左上角 #800*330               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("left_wave1")
            return frame
        elif move == 'left_wave2':
            clap = cv2.imread("images/wave2.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_left_wave2-img_width/2), int(y_left_wave2-img_height/2)#xy是圖的左上角 #900*400               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("left_wave2")
            return frame    
        elif move == 'left_wave3':
            clap = cv2.imread("images/wave3.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_left_wave3-img_width/2), int(y_left_wave3-img_height/2)#xy是圖的左上角 #930*420               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("left_wave3")
            return frame  
        elif move == 'left_wave4':
            clap = cv2.imread("images/wave4.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_left_wave4-img_width/2), int(y_left_wave4-img_height/2)#xy是圖的左上角 #1040*480               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("left_wave4")
            return frame
        elif move == 'left_wave5':
            clap = cv2.imread("images/wave5.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_left_wave5-img_width/2), int(y_left_wave5-img_height/2)#xy是圖的左上角 #1150*280               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("left_wave5")
            return frame
        elif move == 'left_wave6':
            clap = cv2.imread("images/wave6.png")        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_left_wave6-img_width/2), int(y_left_wave6-img_height/2)#xy是圖的左上角 #1040*480               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("left_wave6")
            return frame
        elif move == 'right_wave1':
            clap = cv2.imread("images/wave1.png")
            clap = cv2.flip(clap,1)        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_right_wave1-img_width/2), int(y_right_wave1-img_height/2)#xy是圖的左上角 #800*330               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("right_wave1")
            clap1 = cv2.imread("images/wave6.png")        
            clap1 = cv2.resize( clap1, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height1, img_width1, _ = clap1.shape                           
            clap_gray1 = cv2.cvtColor(clap1, cv2.COLOR_BGR2GRAY)            
            _, clap_mask1 = cv2.threshold(clap_gray1, 25, 255, cv2.THRESH_BINARY_INV)
            x1, y1 = int(x_left_wave6-img_width1/2), int(y_left_wave6-img_height1/2)#xy是圖的左上角 #1040*480               
            clap_area1 = frame[y1: y1+img_height1, x1: x1+img_width1]            
            clap_area_no_clap1 = cv2.bitwise_and(clap_area1, clap_area1, mask=clap_mask1)            
            final_clap1 = cv2.add(clap_area_no_clap1, clap1)            
            frame[y1: y1+img_height1, x1: x1+img_width1] = final_clap1
            print("left_wave6")
            return frame        
        elif move == 'right_wave2':
            clap = cv2.imread("images/wave2.png")
            clap = cv2.flip(clap,1)        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_right_wave2-img_width/2), int(y_right_wave2-img_height/2)#xy是圖的左上角 #700*490               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("right_wave2")
            clap1 = cv2.imread("images/wave6.png")        
            clap1 = cv2.resize( clap1, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height1, img_width1, _ = clap1.shape                           
            clap_gray1 = cv2.cvtColor(clap1, cv2.COLOR_BGR2GRAY)            
            _, clap_mask1 = cv2.threshold(clap_gray1, 25, 255, cv2.THRESH_BINARY_INV)
            x1, y1 = int(x_left_wave6-img_width1/2), int(y_left_wave6-img_height1/2)#xy是圖的左上角 #1040*480               
            clap_area1 = frame[y1: y1+img_height1, x1: x1+img_width1]            
            clap_area_no_clap1 = cv2.bitwise_and(clap_area1, clap_area1, mask=clap_mask1)            
            final_clap1 = cv2.add(clap_area_no_clap1, clap1)            
            frame[y1: y1+img_height1, x1: x1+img_width1] = final_clap1
            print("left_wave6")
            return frame    
        elif move == 'right_wave3':
            clap = cv2.imread("images/wave3.png")
            clap = cv2.flip(clap,1)        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_right_wave3-img_width/2), int(y_right_wave3-img_height/2)#xy是圖的左上角 #650*550               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("right_wave3")
            clap1 = cv2.imread("images/wave6.png")        
            clap1 = cv2.resize( clap1, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height1, img_width1, _ = clap1.shape                           
            clap_gray1 = cv2.cvtColor(clap1, cv2.COLOR_BGR2GRAY)            
            _, clap_mask1 = cv2.threshold(clap_gray1, 25, 255, cv2.THRESH_BINARY_INV)
            x1, y1 = int(x_left_wave6-img_width1/2), int(y_left_wave6-img_height1/2)#xy是圖的左上角 #1040*480               
            clap_area1 = frame[y1: y1+img_height1, x1: x1+img_width1]            
            clap_area_no_clap1 = cv2.bitwise_and(clap_area1, clap_area1, mask=clap_mask1)            
            final_clap1 = cv2.add(clap_area_no_clap1, clap1)            
            frame[y1: y1+img_height1, x1: x1+img_width1] = final_clap1
            print("left_wave6")
            return frame  
        elif move == 'right_wave4':
            clap = cv2.imread("images/wave4.png")
            clap = cv2.flip(clap,1)        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_right_wave4-img_width/2), int(y_right_wave4-img_height/2)#xy是圖的左上角 #560*480               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("right_wave4")
            clap1 = cv2.imread("images/wave6.png")        
            clap1 = cv2.resize( clap1, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height1, img_width1, _ = clap1.shape                           
            clap_gray1 = cv2.cvtColor(clap1, cv2.COLOR_BGR2GRAY)            
            _, clap_mask1 = cv2.threshold(clap_gray1, 25, 255, cv2.THRESH_BINARY_INV)
            x1, y1 = int(x_left_wave6-img_width1/2), int(y_left_wave6-img_height1/2)#xy是圖的左上角 #1040*480               
            clap_area1 = frame[y1: y1+img_height1, x1: x1+img_width1]            
            clap_area_no_clap1 = cv2.bitwise_and(clap_area1, clap_area1, mask=clap_mask1)            
            final_clap1 = cv2.add(clap_area_no_clap1, clap1)            
            frame[y1: y1+img_height1, x1: x1+img_width1] = final_clap1
            print("left_wave6")
            return frame
        elif move == 'right_wave5':
            clap = cv2.imread("images/wave5.png")
            clap = cv2.flip(clap,1)        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_right_wave5-img_width/2), int(y_right_wave5-img_height/2)#xy是圖的左上角 #450*280               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("right_wave5")
            clap1 = cv2.imread("images/wave6.png")        
            clap1 = cv2.resize( clap1, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height1, img_width1, _ = clap1.shape                           
            clap_gray1 = cv2.cvtColor(clap1, cv2.COLOR_BGR2GRAY)            
            _, clap_mask1 = cv2.threshold(clap_gray1, 25, 255, cv2.THRESH_BINARY_INV)
            x1, y1 = int(x_left_wave6-img_width1/2), int(y_left_wave6-img_height1/2)#xy是圖的左上角 #1040*480               
            clap_area1 = frame[y1: y1+img_height1, x1: x1+img_width1]            
            clap_area_no_clap1 = cv2.bitwise_and(clap_area1, clap_area1, mask=clap_mask1)            
            final_clap1 = cv2.add(clap_area_no_clap1, clap1)            
            frame[y1: y1+img_height1, x1: x1+img_width1] = final_clap1
            print("left_wave6")
            return frame
        elif move == 'right_wave6':
            clap = cv2.imread("images/wave6.png")
            clap = cv2.flip(clap,1)        
            clap = cv2.resize( clap, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(x_right_wave6-img_width/2), int(y_right_wave6-img_height/2)#xy是圖的左上角 #490*125               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("right_wave6")
            clap1 = cv2.imread("images/wave6.png")        
            clap1 = cv2.resize( clap1, (250, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height1, img_width1, _ = clap1.shape                           
            clap_gray1 = cv2.cvtColor(clap1, cv2.COLOR_BGR2GRAY)            
            _, clap_mask1 = cv2.threshold(clap_gray1, 25, 255, cv2.THRESH_BINARY_INV)
            x1, y1 = int(x_left_wave6-img_width1/2), int(y_left_wave6-img_height1/2)#xy是圖的左上角 #1040*480               
            clap_area1 = frame[y1: y1+img_height1, x1: x1+img_width1]            
            clap_area_no_clap1 = cv2.bitwise_and(clap_area1, clap_area1, mask=clap_mask1)            
            final_clap1 = cv2.add(clap_area_no_clap1, clap1)            
            frame[y1: y1+img_height1, x1: x1+img_width1] = final_clap1
            print("left_wave6")
            return frame



        elif move == 'open':
            clap = cv2.imread("images/open.png")           
            clap = cv2.resize( clap, (820, 250),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(800-img_width/2), int(220-img_height/2)#xy是圖的左上角 #490*125               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("open")
            return frame
        elif move == 'open_left1':
            clap = cv2.imread("images/open_left1.png")           
            clap = cv2.resize( clap, (750, 380),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(850-img_width/2), int(300-img_height/2)              
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("open_left1")
            return frame
        elif move == 'open_left':
            clap = cv2.imread("images/open_left.png")           
            clap = cv2.resize( clap, (670, 565),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(900-img_width/2), int(300-img_height/2)               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("open_left")
            return frame    
        elif move == 'open_right1':
            clap = cv2.imread("images/open_right1.png")           
            clap = cv2.resize( clap, (750, 380),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(750-img_width/2), int(300-img_height/2)               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("open_right1")
            return frame
        elif move == 'open_right':
            clap = cv2.imread("images/open_right.png")           
            clap = cv2.resize( clap, (670, 565),0,fx=1,fy=1,interpolation=cv2.INTER_AREA)
            img_height, img_width, _ = clap.shape                           
            clap_gray = cv2.cvtColor(clap, cv2.COLOR_BGR2GRAY)            
            _, clap_mask = cv2.threshold(clap_gray, 25, 255, cv2.THRESH_BINARY_INV)
            x, y = int(700-img_width/2), int(300-img_height/2)               
            clap_area = frame[y: y+img_height, x: x+img_width]            
            clap_area_no_clap = cv2.bitwise_and(clap_area, clap_area, mask=clap_mask)            
            final_clap = cv2.add(clap_area_no_clap, clap)            
            frame[y: y+img_height, x: x+img_width] = final_clap
            print("open_right")
            return frame




    ## drawing body
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    ## setting the video source
    cap = cv2.VideoCapture(0)  # webcam
    w = 1600
    h = 960
    cap.set(3, w)  # 3:width
    cap.set(4, h)  # 4:height

    ## setting the sound source
    file = f"sounds/HandClap1m1.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    soundon = 0

    ## setting the start_time of mstimer
    start_time=time.time()
    delay=0
    counter=0

    clap=0
    lh_and_ra=0
    rh_and_la=0
    move=""
    # total_counter=0
    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                      min_tracking_confidence=0.8) as pose:

        # counter = 0  # movement of exercise
        # status = True  # state of move
        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)

            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
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
                if soundon==0 :
                    start_time=time.time()
                    pygame.mixer.music.play()
                    soundon = 1
                
            
                print(mstimer(start_time))

                status=False
                if 4.862+delay < mstimer(start_time) < 5.462+delay:
                    cv2.putText(frame, "3", (1400, 150),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 170, 255), 20, cv2.LINE_AA)
                elif 5.720+delay < mstimer(start_time) < 6.320+delay:
                    cv2.putText(frame, "2", (1400, 150),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 170, 255), 20, cv2.LINE_AA)
                elif 6.578+delay < mstimer(start_time) < 7.178+delay:
                    cv2.putText(frame, "1", (1400, 150),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 170, 255), 20, cv2.LINE_AA)
                elif 7.436+delay < mstimer(start_time) < 8.036+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 8.292+delay < mstimer(start_time) < 8.892+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 9.152+delay < mstimer(start_time) < 9.752+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 10.006+delay < mstimer(start_time) < 10.606+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 10.867+delay < mstimer(start_time) < 11.467+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 11.725+delay < mstimer(start_time) < 12.325+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 12.578+delay < mstimer(start_time) < 13.178+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")                
                elif 14.287+delay < mstimer(start_time) < 14.887+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 15.149+delay < mstimer(start_time) < 15.749+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 16.001+delay < mstimer(start_time) < 16.601+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 16.851+delay < mstimer(start_time) < 17.451+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 17.731+delay < mstimer(start_time) < 18.331+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 18.580+delay < mstimer(start_time) < 19.180+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 19.436+delay < mstimer(start_time) < 20.036+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 20.287+delay < mstimer(start_time) < 20.887+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 21.159+delay < mstimer(start_time) < 21.759+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 22.012+delay < mstimer(start_time) < 22.612+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 22.874+delay < mstimer(start_time) < 23.474+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 23.717+delay < mstimer(start_time) < 24.317+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 24.594+delay < mstimer(start_time) < 25.190+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 25.443+delay < mstimer(start_time) < 26.043+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 26.298+delay < mstimer(start_time) < 26.898+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")
                elif 27.146+delay < mstimer(start_time) < 27.746+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 29.056+delay < mstimer(start_time) <= 29.284+delay:
                    Move(frame,"say_no")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no")
                elif 29.285+delay <= mstimer(start_time) <= 29.785+delay:
                    Move(frame,"say_no1")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no1")
                elif 29.786+delay <= mstimer(start_time) <= 30.229+delay:
                    Move(frame,"say_no")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no")
                elif 30.230+delay <= mstimer(start_time) < 30.767+delay:
                    Move(frame,"say_no1")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no1")
                elif 31.126+delay < mstimer(start_time) < 31.486+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 31.569+delay < mstimer(start_time) < 31.929+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 32.003+delay < mstimer(start_time) < 32.363+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 32.409+delay < mstimer(start_time) < 32.769+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 32.814+delay < mstimer(start_time) < 33.242+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 33.245+delay < mstimer(start_time) < 33.673+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 33.674+delay < mstimer(start_time) < 34.099+delay:
                    Move(frame,"clap")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 34.548+delay < mstimer(start_time) < 34.948+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 34.972+delay < mstimer(start_time) < 35.372+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 35.402+delay < mstimer(start_time) < 35.802+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 35.826+delay < mstimer(start_time) < 36.226+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 36.286+delay < mstimer(start_time) < 36.646+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 36.703+delay < mstimer(start_time) < 37.063+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 37.137+delay < mstimer(start_time) < 37.497+delay:
                    Move(frame,"clap")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 38.310+delay < mstimer(start_time) < 38.910+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 39.161+delay < mstimer(start_time) < 39.761+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 40.016+delay < mstimer(start_time) < 40.616+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 40.865+delay < mstimer(start_time) < 41.465+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 41.730+delay < mstimer(start_time) < 42.330+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 42.580+delay < mstimer(start_time) < 43.180+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 43.441+delay < mstimer(start_time) < 44.041+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 44.296+delay < mstimer(start_time) < 44.896+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 45.160+delay < mstimer(start_time) < 45.760+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 46.012+delay < mstimer(start_time) < 46.612+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 46.867+delay < mstimer(start_time) < 47.467+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 47.724+delay < mstimer(start_time) < 48.324+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 48.579+delay < mstimer(start_time) < 49.179+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 49.434+delay < mstimer(start_time) < 50.034+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")
                elif 50.296+delay < mstimer(start_time) < 50.896+delay:
                    Move(frame,"lh_and_ra")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"lh_and_ra")                
                elif 51.150+delay < mstimer(start_time) < 51.750+delay:
                    Move(frame,"rh_and_la")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"rh_and_la")            
                elif 53.066+delay < mstimer(start_time) < 53.310+delay:
                    Move(frame,"say_no")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no")
                elif 53.311+delay < mstimer(start_time) <= 53.810+delay:
                    Move(frame,"say_no1")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no1")
                elif 53.811+delay <= mstimer(start_time) <= 54.221+delay:
                    Move(frame,"say_no")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no")
                elif 54.222+delay <= mstimer(start_time) < 54.754+delay:
                    Move(frame,"say_no1")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"say_no1")            
                elif 55.004+delay < mstimer(start_time) < 55.484+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 55.557+delay < mstimer(start_time) < 55.917+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 55.999+delay < mstimer(start_time) < 56.359+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 56.411+delay < mstimer(start_time) <= 56.860+delay:
                    Move(frame,"raise_hands_left")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_left")
                elif 56.861+delay <= mstimer(start_time) <= 57.300+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 57.301+delay < mstimer(start_time) <= 57.720+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 57.721+delay <= mstimer(start_time) < 58.184+delay:
                    Move(frame,"clap")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 58.439+delay < mstimer(start_time) < 58.919+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 58.990+delay < mstimer(start_time) < 59.350+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 59.426+delay < mstimer(start_time) < 59.756+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")
                elif 59.741+delay < mstimer(start_time) <= 60.201+delay:
                    Move(frame,"raise_hands_right")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"raise_hands_right")                
                elif 60.301+delay <= mstimer(start_time) <= 60.725+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 60.726+delay <= mstimer(start_time) <= 61.100+delay:
                    Move(frame,"clap")
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"clap")
                elif 61.101+delay <= mstimer(start_time) <= 62.483+delay:
                    Move(frame,"left_wave1")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"left_wave1")
                elif 62.484+delay <= mstimer(start_time) <= 62.686+delay:
                    Move(frame,"left_wave2")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"left_wave2")
                elif 62.687+delay <= mstimer(start_time) < 62.889+delay:
                    Move(frame,"left_wave3")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"left_wave3")  
                elif 62.890+delay <= mstimer(start_time) <= 63.092+delay:
                    Move(frame,"left_wave4")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"left_wave4")
                elif 63.093+delay <= mstimer(start_time) <= 63.295+delay:
                    Move(frame,"left_wave5")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"left_wave5")      
                elif 63.296+delay <= mstimer(start_time) <= 63.500+delay:
                    Move(frame,"left_wave6")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"left_wave6")
                elif 63.501+delay <= mstimer(start_time) <= 63.711+delay:
                    Move(frame,"right_wave1")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"right_wave1")
                elif 63.712+delay <= mstimer(start_time) <= 63.923+delay:
                    Move(frame,"right_wave2")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"right_wave2")
                elif 63.924+delay <= mstimer(start_time) <= 64.134+delay:
                    Move(frame,"right_wave3")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"right_wave3")  
                elif 64.135+delay <= mstimer(start_time) <= 64.346+delay:
                    Move(frame,"right_wave4")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"right_wave4")
                elif 64.347+delay <= mstimer(start_time) <= 64.557+delay:
                    Move(frame,"right_wave5")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"right_wave5")      
                elif 64.558+delay <= mstimer(start_time) <= 65.300+delay:
                    Move(frame,"right_wave6")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"right_wave6")



                elif 65.302+delay <= mstimer(start_time) <= 87.560+delay:
                    Move(frame,"open")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"open")

                elif 88+delay <= mstimer(start_time) <= 95+delay:
                    Move(frame,"open_left1")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"open_left1")

                elif 96+delay <= mstimer(start_time) <= 101+delay:
                    Move(frame,"open_left")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"open_left")

                elif 102+delay <= mstimer(start_time) <= 107+delay:
                    Move(frame,"open_right1")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"open_right1")

                elif 108+delay <= mstimer(start_time) <= 113+delay:
                    Move(frame,"open_right")                
                    counter, status, move = TypeOfDance(landmarks).hand_clap(counter, status,"open_right")

                #print("move ok")

                #counter, status = TypeOfExercise(landmarks).calculate_exercise(
                #    args["exercise_type"], counter, status)
                # RRR=detection_body_parts(landmarks)
                # print(RRR)

            except:
                pass 
            ## render detections (for landmarks)
            # Draw pose landmarks on the image.
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
                #----clap------------------------------------------------------------------------------
                # x_clap_position=800#拍手圖出現位置
                # y_clap_position=330
                x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                x_RIGHT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].x)
                y_RIGHT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_INDEX'].value].y)
                # distance_of_l_INDEX_r_INDEX = int(abs(((x_LEFT_INDEX-x_RIGHT_INDEX)**2+(y_LEFT_INDEX-y_RIGHT_INDEX)**2)**0.5))
                # x_center_of_l_INDEX_r_INDEX=(x_LEFT_INDEX+x_RIGHT_INDEX)/2
                # y_center_of_l_INDEX_r_INDEX=(y_LEFT_INDEX+y_RIGHT_INDEX)/2
                # distance_of_lr_INDEX_clap_position=int(abs(((x_clap_position-x_center_of_l_INDEX_r_INDEX)**2+(y_clap_position-y_center_of_l_INDEX_r_INDEX)**2)**0.5))
                # print(f'distance_of_l_INDEX_r_INDEX: {distance_of_l_INDEX_r_INDEX}')
                # print(f'distance_of_lr_INDEX_clap_position: {distance_of_lr_INDEX_clap_position}')
                # #---lh_and_ra-----------------------------------------------------------------------------------------------
                # x_lh_and_ra_position=900
                # y_lh_and_ra_position=660            
                # x_LEFT_INDEX = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].x)
                # y_LEFT_INDEX = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_INDEX'].value].y)
                # x_RIGHT_ANKLE = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ANKLE'].value].x)
                # y_RIGHT_ANKLE = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ANKLE'].value].y)
                # distance_of_l_INDEX_r_ANKLE = int(abs(((x_LEFT_INDEX-x_RIGHT_ANKLE)**2+(y_LEFT_INDEX-y_RIGHT_ANKLE)**2)**0.5))
                # x_center_of_l_INDEX_r_ANKLE=(x_LEFT_INDEX+x_RIGHT_ANKLE)/2
                # y_center_of_l_INDEX_r_ANKLE=(y_LEFT_INDEX+y_RIGHT_ANKLE)/2            
                # distance_of_lh_and_ra_position=int(abs(((x_lh_and_ra_position-x_center_of_l_INDEX_r_ANKLE)**2+(y_lh_and_ra_position-y_center_of_l_INDEX_r_ANKLE)**2)**0.5))
                # print(f'distance_of_l_INDEX_r_ANKLE: {distance_of_l_INDEX_r_ANKLE}')
                # print(f'distance_of_lh_and_ra_position: {distance_of_lh_and_ra_position}')

                #-----------------------------------------------------------------------------------------------
                cv2.putText(frame, str(f'{x_LEFT_INDEX},{y_LEFT_INDEX}'), (x_LEFT_INDEX+20, y_LEFT_INDEX+20),cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
                #print(f'LEFT_INDEX[{x_LEFT_INDEX},{y_LEFT_INDEX}]')            
                cv2.putText(frame, str(f'{x_RIGHT_INDEX},{y_RIGHT_INDEX}'), (x_RIGHT_INDEX+20, y_RIGHT_INDEX+20),cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)            


                cv2.putText(frame, "Move : " + str(move), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 255), 2, cv2.LINE_AA)             
                cv2.putText(frame, "Counter : " + str(int(counter)), (10, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 255), 2, cv2.LINE_AA)             
                cv2.putText(frame, "Status : " + str(status), (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 255), 2, cv2.LINE_AA)            
                cv2.putText(frame, "Time : " + str(timer(start_time)), (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 255), 2, cv2.LINE_AA)
                cv2.line(frame,(0,125),(1600,125),(150,150,150),3)
                cv2.line(frame,(0,150),(1600,150),(150,150,150),3)
                cv2.line(frame,(0,420),(1600,420),(150,150,150),3)
                cv2.line(frame,(0,690),(1600,690),(150,150,150),3)
                cv2.line(frame,(900,0),(900,960),(150,150,150),3)
                cv2.line(frame,(700,0),(700,960),(150,150,150),3)

                cv2.circle(frame,(810,610),100,(150,150,150), 3)
                cv2.circle(frame,(790,610),100,(150,150,150), 3)

                cv2.circle(frame,(900,620),100,(150,150,150), 3)
                cv2.circle(frame,(700,620),100,(150,150,150), 3)

                cv2.circle(frame,(1070,590),100,(150,150,150), 3)
                cv2.circle(frame,(530,590),100,(150,150,150), 3)

                cv2.circle(frame,(1110,400),100,(150,150,150), 3)
                cv2.circle(frame,(490,400),100,(150,150,150), 3)

                cv2.circle(frame,(1100,220),100,(150,150,150), 3)
                cv2.circle(frame,(500,220),100,(150,150,150), 3)

                #print(222)
            except:
                pass
        
            if timer(start_time) == "00:01:04":
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
                f.write(f"{game_status},{game_type},{counter},{timer(start_time)},{soundon}"+'\n')
            # 生成二進為圖檔    
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


if __name__=='__main__':
    game3_frames()
    

    
    
