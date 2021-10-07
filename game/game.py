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
                counter += 1
                status = True
        elif env[5]=='left_knee':
            left_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
            left_knee = [left_knee[0]*env[0],left_knee[1]*env[1],left_knee[2]]
            if ((left_knee[0]-env[2])**2+(left_knee[1]-env[3])**2 < env[4]**2) and (left_knee[2]>0.6):
                counter += 1
                status = True
        elif env[5]=='right_hand':
            right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
            right_hand = [right_hand[0]*env[0],right_hand[1]*env[1],right_hand[2]]
            if ((right_hand[0]-env[2])**2+(right_hand[1]-env[3])**2 < env[4]**2) and (right_hand[2]>0.6):
                counter += 1
                status = True
        elif env[5]=='right_knee':
            right_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
            right_knee = [right_knee[0]*env[0],right_knee[1]*env[1],right_knee[2]]
            if ((right_knee[0]-env[2])**2+(right_knee[1]-env[3])**2 < env[4]**2) and (right_knee[2]>0.6):
                counter += 1
                status = True
    
        return [counter, status]


    def calculate_exercise(self, game_type, counter, status, env):
        if game_type == "game1":
            counter, status = TypeOfMove(self.landmarks).game1(
                counter, status, env)

        return [counter, status]


#定義環境
class Circle():
    def __init__(self,start=(55,35),length=(1400,500),max_r = 120 ,prob=0.95,position='left_hand'):
        self.status = True  # state of move
        self.add_circle = 0
        self.max_r = max_r
        self.radius = self.max_r
        self.cx = 0
        self.cy = 0
        self.start = start
        self.length = length
        self.prob = prob
        self.position = position
    def play(self,status):
        self.status = status
        if self.status == True:
            self.add_circle = 0
            self.radius = self.max_r
        if self.add_circle == 0:
            self.add_circle = random.random()
            self.cx = self.start[0]+int(self.length[0]*random.random())            
            self.cy = self.start[1]+int(self.length[1]*random.random())
        if self.add_circle > self.prob:
            self.status = False
            self.radius = self.radius-1
            if self.radius>0 and not self.status:
                return self.status,self.cx,self.cy,self.radius,self.position
            else:
                self.add_circle = 0
                self.radius = self.max_r
                self.status = True
                return self.status,0,0,0,self.position
        else:
            self.add_circle = 0
            return self.status,0,0,0,self.position

def game_start(game_type):
    if game_type == "game1":
        lu = Circle(start=(100,35),length=(700,400),position='left_hand')
        lb = Circle(start=(100,600),length=(700,100),position='left_knee')
        ru = Circle(start=(800,35),length=(600,400),position='right_hand')
        rb = Circle(start=(800,600),length=(600,100),position='right_knee')
        return [lu,lb,ru,rb]

def game_play(game_type,env_list):
    if game_type == "game1":
        play_list = []
        for i in env_list:
            play_list.append(i.play(i.status))
        return play_list

def game_plot(game_type,frame,env_coordinate):
    if game_type == "game1":
        color={'left_hand' :(80,80,180),
               'right_hand':(180,80,80),
               'left_knee' :(80,180,80),
               'right_knee':(0,180,255)}
        for env in env_coordinate:
            if not env[0]:
                cv2.circle(frame,(env[1],env[2]),int(env[3]),color[env[4]],-1)
        return frame



