'''
This file is originally from "Sports With AI" https://github.com/Furkan-Gulsen/Sport-With-AI/blob/main/types_of_exercise.py
'''
import numpy as np
from body_part_angle import BodyPartAngle
from utils import *

import random
import cv2


#定義行為
class TypeOfMove(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def game1(self, counter, status, env):
        if env[5]=='left_hand':
            left_hand = detection_body_part(self.landmarks, "RIGHT_INDEX")
            left_hand = [left_hand[0]*env[0],left_hand[1]*env[1],left_hand[2]]
            if ((left_hand[0]-env[2])**2+(left_hand[1]-env[3])**2 < env[4]**2) and (left_hand[2]>0.6):
                counter += 100
                status = True
        # elif env[5]=='left_knee':
        #     left_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        #     left_knee = [left_knee[0]*env[0],left_knee[1]*env[1],left_knee[2]]
        #     if ((left_knee[0]-env[2])**2+(left_knee[1]-env[3])**2 < env[4]**2) and (left_knee[2]>0.6):
        #         counter += 100
        #         status = True
        elif env[5]=='right_hand':
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            right_hand = [right_hand[0]*env[0],right_hand[1]*env[1],right_hand[2]]
            if ((right_hand[0]-env[2])**2+(right_hand[1]-env[3])**2 < env[4]**2) and (right_hand[2]>0.6):
                counter += 100
                status = True
        # elif env[5]=='right_knee':
        #     right_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        #     right_knee = [right_knee[0]*env[0],right_knee[1]*env[1],right_knee[2]]
        #     if ((right_knee[0]-env[2])**2+(right_knee[1]-env[3])**2 < env[4]**2) and (right_knee[2]>0.6):
        #         counter += 100
        #         status = True
        elif env[5]=='left_dodge' or env[5]=='right_dodge':
            right_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
            left_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
            right_hip = detection_body_part(self.landmarks, "RIGHT_HIP")
            left_hip = detection_body_part(self.landmarks, "LEFT_HIP")
            all_part = [right_shoulder,left_shoulder,right_hip,left_hip]
            check = 0
            for part in all_part:
                if 0<(part[0]*env[0]-env[2][0])<env[3][0] and 0<(part[1]*env[1]-env[2][1])<env[3][1] and (part[2]>0.6):
                    check += 1
            if check == 4:
                counter += 100
                status = True
            print(check)
        return [counter, status]
    
    def game2(self, counter, status, env):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")

        if status:
            if left_knee[0] > right_knee[0]:
                counter += 1
                status = False                

        else:
            if left_knee[0] < right_knee[0]:
                counter += 1
                status = True

        return [counter, status]
    

    def calculate_exercise(self, game_type, counter, status, env=None):
        if game_type == "game1":
            counter, status = TypeOfMove(self.landmarks).game1(
                counter, status, env)
        elif game_type == "game2":
            counter, status = TypeOfMove(self.landmarks).game2(
                counter, status, env)

        return [counter, status]


#定義環境
#圓圈圈
class Circle():
    def __init__(self,start=(55,35),length=(800,400),max_r = 130 ,time = 0 ,position='left_hand'):
        self.status = True  # state of move
        self.max_r = max_r
        self.radius = 0
        self.cx = 0
        self.cy = 0
        self.start = start
        self.length = length
        self.time = time
        self.position = position

    def play(self,status,time):
        self.status = status
        if 0<(time-self.time)<0.2 and self.status:
            self.status = False
            self.cx = self.start[0]+int(self.length[0]*random.random())            
            self.cy = self.start[1]+int(self.length[1]*random.random())
            self.radius = self.max_r
        if self.radius>60 and not self.status:
            self.radius = self.radius-6
            return self.status,self.cx,self.cy,self.radius,self.position
        else:
            self.add_circle = 0
            self.radius = self.max_r
            self.status = True
            return self.status,0,0,0,self.position

class Rectangle():
    def __init__(self,start=(500,35),length=(512,300) ,time = 0 ,position='left_dodge'):
        self.status = True  # state of move
        self.start = start
        self.length = length
        self.time = time
        self.opaque = 0  #透明度
        self.position = position

    def play(self,status,time): 
        self.status = status
        if 0<(time-self.time)<0.2 and self.status: 
            self.status = False
            self.start = (self.start[0]+int(self.length[0]*random.random()*.1),self.start[1]+int(self.length[1]*random.random()*.1)) #計算XY
            self.length = (self.length[0],self.length[1])
        if (time-self.time)<3 and not self.status:
            self.opaque = 1 # min(1,self.opaque+0.2) #透明度調整最大值為1
            return self.status,self.start,self.length,self.opaque,self.position
        else:
            self.status = True
            self.opaque = 0
            return self.status,(0,0),(0,0),0,self.position
        

def game_start(game_type):
    if game_type == "game1":
        tick = [7.496,8.188,9.035,9.916,11.348,12.795,13.661,15.186,16.552,18.030,19.257,20.227,20.745,22.424,22.683,23.199,23.615,24.557,
                25.493,27.002,27.952,30.570,32.040,34.211,34.915,35.672,37.565,38.639,39.000,39.643,40.367,41.570,42.283,43.221,43.901,44.118,
                43.901,44.118,45.125,46.000,47.004,47.798,49.065,49.759,50.728,51.176,51.394,51.618]
        # tick = 0.469 #60/bpm數字
        rand_list = []
        # for i in range(len(tick)):
        for g in tick:
            # print(g)

## =====雨軒修改: 針對1280*960螢幕比例重新規劃出現位置=====
            if random.random()>0.3:
                choice = random.sample([[(800,320),(200,150),'left_hand'],[(300,320),(200,150),'right_hand']],k=1)
                ok = Circle(start=choice[0][0],length=choice[0][1],time = g-0.05,position=choice[0][2]) #choice 取陣列[(860,240),(120,150),'left_hand']後取段               
                rand_list.append(ok)
                continue

            else:
                choice = random.sample([[(300,135),(300,600),'left_dodge'],[(650,135),(300,600),'right_dodge']],k=1)
                    # print(choise_list) 
                ok = Rectangle(start=choice[0][0],length=choice[0][1],time = g-0.05,position=choice[0][2])
                    # if rand_list[-1].position != ok.position:
                    #     ok = Rectangle(start=choice[0][0],length=choice[0][1],time = g-0.05,position=choice[0][2])
                # 必免重覆出現兩次同方向閃躲
                if rand_list != [] and rand_list[-1].position == ok.position:
                    choice = random.sample([[(800,320),(200,150),'left_hand'],[(500,320),(-200,150),'right_hand']],k=1)
                    ok = Circle(start=choice[0][0],length=choice[0][1],time = g-0.05,position=choice[0][2]) #choice 取陣列[(860,240),(120,150),'left_hand']後取段               
                    rand_list.append(ok)
                    continue
                else:
                    rand_list.append(ok)
                    continue
## =====雨軒修改結束=====
            # for q in range(1,20):
            #     choice = random.sample([[(900,240),(200,180),'left_hand'],[(650,240),(200,180),'right_hand']],k=1)
            #     ok = Circle(start=choice[0][0],length=choice[0][1],time = (88+q)*g+0.1,position=choice[0][2])
            #     rand_list.append(ok)
        return rand_list
def game_play(game_type,env_list,time):
    if game_type == "game1":
        play_list = []
        for i in env_list:
            play_list.append(i.play(i.status,time))
        return play_list

def game_plot(game_type,frame,env_coordinate):
    if game_type == "game1":
        color={'left_hand' :(80,80,180),
               'right_hand':(180,80,80),
               'left_knee' :(80,180,80),
               'right_knee':(0,180,255),
               'left_dodge':(55,180,55),
               'right_dodge':(180,180,55)}
        for env in env_coordinate:
            if not env[0]:
                if env[4] in ['left_hand','right_hand','left_knee','right_knee']:
                    punch = cv2.imread(f'images/{env[4]}.png',-1)
                    y1, y2 = int(env[2]-0.5*punch.shape[0]), int(env[2] + 0.5*punch.shape[0])
                    x1, x2 = int(env[1]-0.5*punch.shape[1]), int(env[1] + 0.5*punch.shape[1])

                    alpha_s = punch[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (alpha_s * punch[:, :, c] +
                                                alpha_l * frame[y1:y2, x1:x2, c])
                    cv2.circle(frame,(env[1],env[2]),int(env[3]),color[env[4]],2)
                elif env[4] in ['left_dodge','right_dodge']:
                    dodge = cv2.imread(f'images/{env[4]}.png',-1)
                    y1, y2 = env[1][1], env[1][1] + dodge.shape[0]
                    x1, x2 = env[1][0], env[1][0] + dodge.shape[1]

                    alpha_s = dodge[:, :, 3] / 255.0
                    alpha_s = alpha_s * env[3]
                    alpha_l = 1.0 - alpha_s

                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (alpha_s * dodge[:, :, c] +
                                                alpha_l * frame[y1:y2, x1:x2, c])
                    # cv2.rectangle(frame,(int(env[1][0]),int(env[1][1])),(int(env[1][0]+env[2][0]),int(env[1][1]+env[2][1])),color[env[4]],3)
        return frame



