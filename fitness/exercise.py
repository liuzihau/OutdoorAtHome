'''
This file is originally from "Sports With AI" https://github.com/Furkan-Gulsen/Sport-With-AI/blob/main/types_of_exercise.py
'''
import numpy as np
import math
from body_part_angle import BodyPartAngle
from utils import *

class TypeOfExercise(BodyPartAngle):
    def __init__(self, landmarks):
        super().__init__(landmarks)

    #伏地挺身(側面)(胸)
    def push_up(self, counter, status, hint):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2
        abdomen_angle = self.angle_of_the_abdomen()

        if status:
            if avg_arm_angle <= 90:
                counter += 1
                status = False
        else:
            if avg_arm_angle > 160:
                status = True

        if hint:
            if avg_arm_angle < 160:
                hint= "Contract your abs and tighten your core by pulling your belly button toward your spine."            
            elif avg_arm_angle > 90:
                hint= "Slowly bend elbows , until your elbows are at a 90-degree angle."
            elif abdomen_angle < 160:
                hint= "Keep your body in a straight line from head to toe"        
        return [counter, status, hint]      

    # def push_up_method_2():

    # 引體向上(正面)(背)
    def pull_up(self, counter, status, hint):
        nose = detection_body_part(self.landmarks, "NOSE")        
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        right_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        left_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        avg_shoulder_y = (left_elbow[1] + right_elbow[1]) / 2       

        if status:
            if nose[1] > avg_shoulder_y:
                counter += 1
                status = False
        else:
            if nose[1] < avg_shoulder_y:
                status = True

        if hint:
            if nose[1] < avg_shoulder_y:
                hint= "Leap up and grip the bar with your hands shoulder width apart and your palms facing away from you."
            #若左右手肘向外撐開                
            elif right_elbow[0] < right_wrist[0] & left_elbow[0] > left_wrist[0] :
                hint= "Keep your shoulders back ，focus on your back muscles and core"
            
        return [counter, status, hint]

    # 深蹲squat(側面)(臀腿)
    def squat(self, counter, status, hint):
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2
        abdomen_angle = self.angle_of_the_abdomen()

        if status:
            if avg_leg_angle < 70:
                counter += 1
                status = False
        else:
            if avg_leg_angle > 160:
                status = True
        if hint:
            if avg_leg_angle > 165:
                hint= "Squat slowly and keeping your weight behind your center of gravity"            
            elif avg_leg_angle > 70 and avg_leg_angle <= 90:
                hint= "Squat lower"
            elif abdomen_angle < 60:
                hint= "Chest up and shift the center of gravity backward"  
        
        return [counter, status, hint]

    def walk(self, counter, status, hint):
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
                
        if hint:
            if left_knee[0] > right_knee[0]:
                hint= "left" 
            elif left_knee[0] < right_knee[0]:
                hint= "right"

        return [counter, status, hint]

    #仰臥起坐sit up(側面)(核心)
    def sit_up(self, counter, status, hint):
        angle = self.angle_of_the_abdomen()
        neck_forward_angle = self.angle_of_the_neck_forward()
        if status:
            if angle < 55:
                counter += 1
                status = False
        else:
            if angle > 105:
                status = True

        if hint:
            if neck_forward_angle < 150:
                hint= "Try not to bend your neck"
            
        return [counter, status, hint]
    #抬腿lying leg raises(側面)(核心)
    def lift_leg(self, counter, status, hint):
        abdomen_angle = self.angle_of_the_abdomen()
        neck_forward_angle = self.angle_of_the_neck_forward()
        if status:
            if abdomen_angle < 145:
                counter += 1
                status = False
        else:
            if abdomen_angle > 160:
                status = True

        if hint:
            if abdomen_angle > 175:
                hint= "Try to lift your legs higher "            
            elif abdomen_angle < 95:
                hint= "Try to lift your legs lower"
            elif neck_forward_angle < 165:
                hint= "Avoid forces your neck muscles to bend forward"
            
        return [counter, status, hint]

    # 橋式bridge(側面)(臀)
    def bridge(self, counter, status, hint):
        # 利用挺腰角度判斷
        abdomen_angle = self.angle_of_the_abdomen()
        # 小腿與地面夾角  
        calf_horizon = self.angle_of_the_calf_horizon()      

        # print(angle)
        if status:
            if abdomen_angle > 160:
                counter += 1
                status = False
        else:
            if abdomen_angle < 120:
                status = True

        if hint:
            if calf_horizon > 110 | calf_horizon < 70:
                hint= "Try to put your calves at a 90-degree angle to the ground"            
            elif abdomen_angle > 175:
                hint= " Avoid raising your hips too high"            
                
        return [counter, status, hint]

    # 腳踏車核心 Bicycle crunch(側面)(核心)
    def bicycle(self, counter, status, hint):
        right_knee = detection_body_part(self.landmarks, "RIGHT_KNEE")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        # 肘膝對稱距離計算
        length_relb_lknee = math.hypot(right_elbow[0]-left_knee[0], right_elbow[1]-left_knee[1])
        length_lelb_rknee = math.hypot(left_elbow[0]-right_knee[0], left_elbow[1]-right_knee[1])
        # print(f'Relb_Lknee:{length_relb_lknee}')
        # print(f'Lelb_Rknee:{length_lelb_rknee}')
        neck_forward_angle = self.angle_of_the_neck_forward()
        # 左右各做一下才算數
        if status:
            if length_relb_lknee < 0.15:
                    counter += 0.5
                    status = False
            elif length_lelb_rknee < 0.15:
                    counter += 0.5
                    status = False
        else:
                if length_relb_lknee >= 0.2 and length_lelb_rknee >= 0.2:
                    status = True
        if hint:
            if neck_forward_angle < 100:
                hint= "Avoid forces your neck muscles to bend forward"
        return [counter, status, hint]

    #飛鳥/側平舉 side_lateral_raise(正面)(肩)
    def side_lateral_raise(self, counter, status, hint):
        right_shoulder_angle=self.angle_of_the_right_shoulder()
        left_shoulder_angle=self.angle_of_the_left_shoulder()
        right_elbow = detection_body_part(self.landmarks, "RIGHT_ELBOW")
        right_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        left_elbow = detection_body_part(self.landmarks, "LEFT_ELBOW")
        left_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        
        if status:
            if left_shoulder_angle > 60 or right_shoulder_angle > 60:
                counter += 1
                status = False
        else:
            if left_shoulder_angle < 15 or right_shoulder_angle < 15:
                status = True

        if hint:
            if left_shoulder_angle > 90 or right_shoulder_angle > 90:
                hint= "Stop your shoulders from shrugging"            
            elif right_wrist[1] < right_elbow[1] or left_wrist[1] < left_elbow[1] :
                hint= "Your elbow should higher than the dumbbell"            
                 
        return [counter, status, hint] 

    #臀推 hip thrust (側面)(臀)
    def hip_thrust(self, counter, status, hint):        
        abdomen_angle = self.angle_of_the_abdomen()
        left_leg_angle = self.angle_of_the_right_leg()
        right_leg_angle = self.angle_of_the_left_leg()
        avg_leg_angle = (left_leg_angle + right_leg_angle) // 2        
        
        if status:
            if abdomen_angle > 170:
                counter += 1
                status = False
        else:
            if abdomen_angle < 120:
                status = True

        if hint:
            if abdomen_angle < 180:
                hint= "Try to squeeze your butt glutes "            
            elif abdomen_angle > 170:
                if avg_leg_angle < 80:
                    hint= "Try to put your calves at a 90-degree angle to the ground"            
                
        return [counter, status, hint]
        
    #啞鈴臥推 dumbbell bench press(側面)(肩)
    def dumbbell_bench_press(self, counter, status, hint):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        avg_arm_angle = (left_arm_angle + right_arm_angle) // 2 
        right_shoulder_angle=self.angle_of_the_right_shoulder()
        left_shoulder_angle=self.angle_of_the_left_shoulder()
        avg_shoulder_angle = (right_shoulder_angle + left_shoulder_angle) // 2              
        
        if status:
            if avg_arm_angle > 150:
                counter += 1
                status = False
        else:
            if avg_arm_angle < 90:
                status = True

        if hint:
            if avg_arm_angle > 150: 
                if avg_shoulder_angle < 160:
                    hint= "Try to press up your dumbbells closer"            
            elif avg_arm_angle < 160:
                hint= "Try to press up with straight arms"            
                 
        return [counter, status, hint]

         
    #啞鈴俯身划船 Bent Over Row(側面)(背)
    def bent_over_row(self, counter, status, hint):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        left_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        right_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")       

        if status:
            if left_arm_angle < 90 or right_arm_angle < 90:
                counter += 1
                status = False
        else:
            if left_arm_angle > 160 or right_arm_angle > 160:
                status = True

        if hint:
            if right_arm_angle < 90: 
                if right_shoulder[1] > left_shoulder[1]:
                    hint= "Try to press up your dumbbells back and across to contract your lats and abduction the arm"            
            elif left_arm_angle < 90: 
                if left_shoulder[1] > right_shoulder[1]:
                    hint= "Try to press up your dumbbells back and across to contract your lats and abduction the arm"

        return [counter, status, hint]

    # 二頭彎舉 Biceps Curls(正面)(手臂)
    def biceps_curls(self, counter, status ,hint):
        left_arm_angle = self.angle_of_the_left_arm()
        right_arm_angle = self.angle_of_the_right_arm()
        left_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        right_shoulder = detection_body_part(self.landmarks, "RIGHT_SHOULDER")
        right_wrist = detection_body_part(self.landmarks, "RIGHT_WRIST")
        left_wrist = detection_body_part(self.landmarks, "LEFT_WRIST")
        neck_forward_angle = self.angle_of_the_neck_forward()
        #print(left_arm_angle,right_arm_angle)

        # 兩手都達成才加一分
        if status:
            if left_arm_angle < 70 and right_arm_angle < 70:
                counter += 1
                status = False
            elif left_arm_angle < 70 or right_arm_angle < 70:
                counter += 0.5
                status = False
        else:
            if left_arm_angle >= 160 and right_arm_angle >= 160:
                status = True

        if hint:
            if right_arm_angle < 70 or left_arm_angle < 70: 
                if right_wrist[1] - right_shoulder[1] > 0.1 or  left_wrist[1] - left_shoulder[1] > 0.1:
                    hint= "Activate your core and try to move dumbbell higher"            
            elif neck_forward_angle < 150: 
                hint= "Chest up and face forward"

        return [counter, status, hint]


    def calculate_exercise(self, exercise_type, counter, status, hint):
        if exercise_type == "push-up":
            counter, status , hint= TypeOfExercise(self.landmarks).push_up(
                counter, status, hint)
        elif exercise_type == "pull-up":
            counter, status , hint= TypeOfExercise(self.landmarks).pull_up(
                counter, status, hint)
        elif exercise_type == "squat":
            counter, status ,hint= TypeOfExercise(self.landmarks).squat(
                counter, status, hint)
        elif exercise_type == "walk":
            counter, status , hint= TypeOfExercise(self.landmarks).walk(
                counter, status, hint)
        elif exercise_type == "sit-up":
            counter, status , hint= TypeOfExercise(self.landmarks).sit_up(
                counter, status, hint)
        elif exercise_type == "lift-leg":
            counter, status , hint= TypeOfExercise(self.landmarks).lift_leg(
                counter, status, hint)
        elif exercise_type == "bridge":
            counter, status, hint= TypeOfExercise(self.landmarks).bridge(
                counter, status, hint)
        elif exercise_type == "bicycle":
            counter, status, hint= TypeOfExercise(self.landmarks).bicycle(
                counter, status, hint)
        elif exercise_type == "side_lateral_raise":
            counter, status, hint= TypeOfExercise(self.landmarks).side_lateral_raise(
                counter, status, hint)
        elif exercise_type == "hip_thrust":
            counter, status, hint = TypeOfExercise(self.landmarks).hip_thrust(
                counter, status, hint)
        elif exercise_type == "dumbbell_bench_press":
            counter, status, hint = TypeOfExercise(self.landmarks).dumbbell_bench_press(
                counter, status, hint)
        elif exercise_type == "bent_over_row":
            counter, status, hint = TypeOfExercise(self.landmarks).bent_over_row(
                counter, status, hint)
        elif exercise_type == "biceps_curls":
            counter, status ,hint= TypeOfExercise(self.landmarks).biceps_curls(
                counter, status, hint)

        return [counter, status ,hint]
