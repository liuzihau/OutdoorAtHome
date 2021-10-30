'''
This file is originally from "Sports With AI" https://github.com/Furkan-Gulsen/Sport-With-AI/blob/main/types_of_exercise.py
'''
import numpy as np
from body_part_angle import BodyPartAngle
from utils import *
import math
# import autopy
import pyautogui


class TypeOfControl(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    def control(self, counter, status, hint):
        right_hand = detection_body_part(self.landmarks, "LEFT_INDEX")
        right_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER") 
        left_hand = detection_body_part(self.landmarks, "RIGHT_INDEX")
        left_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")

        # 肩寬
        length_center_x = abs((left_shoulder[0]-right_shoulder[0])*100)

        
        # 右手與右肩距離
        length_rshder_rhand_x = (right_hand[0]-right_shoulder[0])*100
        length_rshder_rhand_y = (right_hand[1]-right_shoulder[1])*100
        length_rshder_rhand = math.hypot(right_hand[0]-right_shoulder[0], right_hand[1]-right_shoulder[1])*100
        # 左手左肩
        length_lshder_lhand_x = (left_hand[0]-left_shoulder[0])*100
        length_lshder_lhand_y = (left_hand[1]-left_shoulder[1])*100         
        length_lshder_lhand = math.hypot(left_hand[0]-left_shoulder[0], left_hand[1]-left_shoulder[1])*100

        
        
        if status and right_shoulder[2]>0.5 and right_hand[2]>0.5:
            if length_rshder_rhand_x > length_center_x*1.2 or length_lshder_lhand_x > length_center_x:
                hint="right"
                # autopy.key.tap(autopy.key.Code.RIGHT_ARROW)
                pyautogui.press('right')
                status = False
                # print(status,hint,length_rshder_rhand)
            elif length_rshder_rhand_x < -length_center_x or length_lshder_lhand_x < -length_center_x*1.2:
                hint="left"
                # autopy.key.tap(autopy.key.Code.LEFT_ARROW)
                pyautogui.press('left')
                status = False
                # print(status,hint,length_rshder_rhand)

            elif length_rshder_rhand_y < -length_center_x*1.5:
                hint="up"
                # autopy.key.tap(autopy.key.Code.UP_ARROW)
                pyautogui.press('up')
                status = False
                # print(status,hint,length_rshder_rhand)
                
            elif length_lshder_lhand_y < -length_center_x*1.5:
                hint="left_up"
                pyautogui.press('down')
                status = False
                # print(status,hint,length_lshder_lhand)


        elif length_rshder_rhand >= length_center_x*1.2 and right_shoulder[2]>0.5 and right_hand[2]>0.5:
            time.sleep(1)
            status = True
            # print(status,length_rshder_rhand)
        else:
            status = False
            # print(status,length_rshder_rhand)

        return [counter, status, hint]

    

    def calculate_exercise(self, exercise_type, counter, status, hint):
        if exercise_type == "control":
            counter, status, hint = TypeOfControl(self.landmarks).control(
                counter, status, hint)
        return [counter, status, hint]
