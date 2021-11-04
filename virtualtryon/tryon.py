'''
This file is originally from "Sports With AI" https://github.com/Furkan-Gulsen/Sport-With-AI/blob/main/types_of_exercise.py
'''
import os
import numpy as np
from body_part_angle import BodyPartAngle
from utils import *
import math
import time
import cv2
import uuid

#定義行為
class TypeOfTry(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def tryon1(self, counter, status, env, w, h,tryon_status,page):
        if 'choice' in env.position or 'test' in env.position:
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            if 0<(right_hand[0]*w-env.start[0])<env.length[0] and 0<(right_hand[1]*h-env.start[1])<env.length[1] and (right_hand[2]>0.6):
                env.choice = True
        elif env.position =='reset':
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            if not env.status and 0<(right_hand[0]*w-env.start[0])<env.length[0] and 0<(right_hand[1]*h-env.start[1])<env.length[1] and (right_hand[2]>0.6):
                env.warm += 1
            else:
                env.warm = 0
            if env.warm >= 12:
                tryon_status -= 1
                env.warm = 0
        elif env.position =='ok':
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            if not env.status and 0<(right_hand[0]*w-env.start[0])<env.length[0] and 0<(right_hand[1]*h-env.start[1])<env.length[1] and (right_hand[2]>0.6):
                env.warm += 1
            else:
                env.warm = 0
            if env.warm >= 12:
                tryon_status += 1
                env.warm = 0
        elif env.position =='next':
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            if not env.status and 0<(right_hand[0]*w-env.start[0])<env.length[0] and 0<(right_hand[1]*h-env.start[1])<env.length[1] and (right_hand[2]>0.6):
                env.warm += 1
            else:
                env.warm = 0
            if env.warm >= 12:
                page += 1
                env.warm = 0
        elif env.position =='prev':
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            if not env.status and 0<(right_hand[0]*w-env.start[0])<env.length[0] and 0<(right_hand[1]*h-env.start[1])<env.length[1] and (right_hand[2]>0.6):
                env.warm += 1
            else:
                env.warm = 0
            if env.warm >= 12:
                page -= 1
                env.warm = 0
        return [counter, status, tryon_status,page]


    def calculate_exercise(self, tryon_type, counter, status, env, w,h ,tryon_status,page):
        if tryon_type == "tryon1":
            counter, status , tryon_status , page= TypeOfTry(self.landmarks).tryon1(
                counter, status, env , w, h, tryon_status , page)
        return [counter, status, tryon_status , page]


#定義環境
class Rectangle():
    def __init__(self,start=(500,35),length=(512,300) ,time = 0 ,position='choice0'):
        self.status = True  # state of move
        self.start = start
        self.length = length
        self.time = time
        self.opaque = 0
        self.position = position
        self.choice = False 
        self.warm = 0 


    def play(self,status,time):
        self.status = status
        if not self.status:
            self.opaque = min(1,self.opaque+0.2)
            return self.status,self.start,self.length,self.opaque,self.position,self.choice,self.warm
        else:
            self.status = True
            self.opaque = 0
            return self.status,(0,0),(0,0),0,self.position,self.choice,self.warm
        

def tryon_start(tryon_type):
    if tryon_type == "tryon1":
        l1 = Rectangle(start=(260,640),length=(240,360),time = 5,position='choice0')
        l2 = Rectangle(start=(360,220),length=(240,360),time = 5,position='choice1')
        r1 = Rectangle(start=(1000,220),length=(240,360),time = 5,position='choice2')
        r2 = Rectangle(start=(1100,640),length=(240,360),time = 5,position='choice3')
        ok = Rectangle(start=(620,160),length=(160,160),time = 5,position='ok')
        reset = Rectangle(start=(820,160),length=(160,160),time = 5,position='reset')
        return [l1,l2,r1,r2,ok,reset]

def add_product(path):
    # PRODUCT_FILE = [f"{path}/{file}"for file in os.listdir(path)]
    files = os.listdir(path)
    p_list = []
    for i,file in enumerate(files):
        name = file.split('.')[0]
        if i % 4 == 0:
            p = Rectangle(start=(300,640),length=(240,360),time = 3,position=f'test-{i}-{name}')
            p_list.append(p)
        elif i % 4 == 1:
            p = Rectangle(start=(400,220),length=(240,360),time = 3,position=f'test-{i}-{name}')
            p_list.append(p)
        elif i % 4 == 2:
            p = Rectangle(start=(1000,220),length=(240,360),time = 3,position=f'test-{i}-{name}')
            p_list.append(p)
        elif i % 4 == 3:
            p = Rectangle(start=(1100,640),length=(240,360),time = 3,position=f'test-{i}-{name}')
            p_list.append(p)
    ok = Rectangle(start=(620,160),length=(160,160),time = 5,position='ok')
    reset = Rectangle(start=(820,160),length=(160,160),time = 5,position='reset')
    next = Rectangle(start=(880,670),length=(160,160),time = 5,position='next')
    prev = Rectangle(start=(560,670),length=(160,160),time = 5,position='prev')
    return p_list,[ok,reset,next,prev]

def tryon_play(tryon_type,env_list,time):
    if tryon_type == "tryon1":
        play_list = []
        for i in env_list:
            play_list.append(i.play(False,time))
        return play_list

def tryon_plot(tryon_type,frame,env_coordinate,path=None):
    if tryon_type == "tryon1":
        color={'choice0':(60,90,200),
               'choice1':(60,90,200),
               'choice2':(60,90,200),
               'choice3':(60,90,200),
               'ok':(180,80,80),
               'reset':(80,180,80)}
        for env in env_coordinate:
            if not env[0]:
                if 'test' in env[4]:
                    filename = env[4].split('-')[2]
                    choice = cv2.imread(f'{path}/{filename}.jpg',-1)
                else:
                    choice = cv2.imread(f'trypose/{env[4]}.png',-1)
                choice = cv2.resize(choice,(env[2][0], env[2][1]), interpolation=cv2.INTER_AREA)
                y1, y2 = env[1][1], env[1][1] + choice.shape[0]
                x1, x2 = env[1][0], env[1][0] + choice.shape[1]
                if choice.shape[2]==4:
                    alpha_s = choice[:, :, 3] / 255.0
                    alpha_s = alpha_s * env[3]
                    alpha_l = 1.0 - alpha_s

                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (alpha_s * choice[:, :, c] +
                                                alpha_l * frame[y1:y2, x1:x2, c])
                else:
                    frame[y1:y2, x1:x2, :] = choice
                if 'choice' in env[4] or 'test' in env[4]:
                    cv2.rectangle(frame,(int(env[1][0]),int(env[1][1])),(int(env[1][0]+env[2][0]),int(env[1][1]+env[2][1])),(60,90,200),3)
                    if env[5]:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,"Choosed!",(int(env[1][0]),int(env[1][1]-8)),font,1,(80,127,255),3,cv2.LINE_AA)
                elif env[4] in ['ok','reset','next','prev']:
                    if env[6]>0:
                        cv2.circle(frame,(int(env[1][0]+0.5* env[2][0]),int(env[1][1]+0.5*env[2][1])),int(180-8*env[6]),(150,150,250),4)

        return frame

def tryon2_plot(tryon_type,frame,env_list,w,h,tryon2start):
    if tryon_type == "tryon1":
        purge = 0
        for i,env in enumerate(env_list):
            if env.choice:
                if env.position in ['choice0','choice1','choice2','choice3']:
                    print(tryon2start)
                    diff = time.time()-tryon2start
                    real_time = 6-diff
                    print(real_time)
                    if 0<real_time<0.1:
                        filename = round(tryon2start)
                        cv2.imwrite(f"tryimages/test{filename}.jpg",frame)
                        tryon2start = time.time()
                        print(tryon2start)
                        purge = 1
                    if real_time>0.2:
                        env.length=(768,1024)
                        choice = cv2.imread(f'trypose/{env.position}.png',-1)
                        y1, y2 = h-1025, h-1025 + 1024
                        x1, x2 = int(w/2-768/2), int(w/2-768/2) + 768
                        alpha_s = choice[:, :, 3] / 255.0
                        alpha_s = alpha_s * 0.7
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            frame[y1:y2, x1:x2, c] = (alpha_s * choice[:, :, c] +
                                                    alpha_l * frame[y1:y2, x1:x2, c])
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,"Please Stand",(20,70),font,2,(80,127,255),3,cv2.LINE_AA)
                        cv2.putText(frame,"in shadow",(250,140),font,2,(80,127,255),3,cv2.LINE_AA)
                        cv2.putText(frame,f"remain: {math.ceil(real_time)}",(1250,70),font,2,(80,127,255),3,cv2.LINE_AA)
                    
            else:
                purge = 1
            break
        if purge==1:
            env_list.pop(0)
        return frame,tryon2start


