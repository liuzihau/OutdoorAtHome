import cv2
import mediapipe as mp
import numpy as np
from game.game import *
import argparse
import pygame
import random
import time
from utils import *
from body_part_angle import BodyPartAngle
## setup agrparse



ap = argparse.ArgumentParser()

ap.add_argument("-t",
                "--game_type",
                type=str,
                help='Type of activity to do',
                required=True)
ap.add_argument("-vs",
                "--video_source",
                type=str,
                help='Type of activity to do',
                required=False)
args = vars(ap.parse_args())
def timer1(start_time):
  time_diff = time.time()-start_time
  return time_diff
file = f"sounds/game2.mp3"
pygame.mixer.init()
pygame.mixer.music.load(file)
game_over = False
soundon = 0
start = 0
vic_time = 0
game_status = 0 # game start
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
#設置遊戲初始環境
start_time = time.time()
pose = mp_pose.Pose(min_detection_confidence=0.8,
                     min_tracking_confidence=0.8)
counter = 0 
hard = 20
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
    # if counter<hard-1:
    #   counter += 0.2
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
      dead_time = time.time()-over_time
      soundon = 2
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
      if start == 0:
        cap = cv2.VideoCapture('images/victory.mp4')
        start +=1
      elif cap.get(cv2.CAP_PROP_POS_FRAMES) <227:
        ret , final_frame = cap.read()
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
        counter, status= TypeOfMove(landmarks).calculate_exercise(args["game_type"], counter, status)
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
    


    with open('game.txt','w+') as f:
                f.write(f"{game_status},{counter},{vic_time}"+'\n')

    cv2.imshow('MediaPipe Selfie Segmentation', final_frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

