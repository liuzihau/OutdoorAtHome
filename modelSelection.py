import cv2
import mediapipe as mp
import numpy as np
from game.game import *
import argparse
import pygame
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

file = f"sounds/game2.mp3"
pygame.mixer.init()
pygame.mixer.music.load(file)
soundon = 0

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
#設置遊戲初始環境
start_time = time.time()
pose = mp_pose.Pose(min_detection_confidence=0.8,
                     min_tracking_confidence=0.8)
counter = 0 
hard = 50

BG_COLOR = cv2.imread("images/004.png")

Orix = int(1533-(893/hard)*counter)
Oriy = int(1150-(670/hard)*counter)
BG_COLOR = cv2.resize(BG_COLOR, (Orix, Oriy), interpolation=cv2.INTER_AREA)
x = int(447-(447/hard)*counter)
y = int(335-(335/hard)*counter)
w = 640
h = 480
BG_COLOR = BG_COLOR[y:y+h, x:x+w]
cap = cv2.VideoCapture(0)
status = True
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
  bg_image = None
  while cap.isOpened():
    if counter<=hard:
      if counter >= hard-10:
        BG_COLOR = cv2.imread("images/003.png")
      else:
        BG_COLOR = cv2.imread("images/004.png")
      Orix = int(1533-(893/hard)*counter)
      Oriy = int(1150-(670/hard)*counter)
      BG_COLOR = cv2.resize(BG_COLOR, (Orix, Oriy), interpolation=cv2.INTER_AREA)
      x = int(447-(447/hard)*counter)
      y = int(335-(335/hard)*counter)
      w = 640
      h = 480
      BG_COLOR = BG_COLOR[y:y+h, x:x+w]
      print(counter)
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
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

      if soundon==0 :
        pygame.mixer.music.play()
        soundon = 1
        start_time = time.time()

      landmarks = results.pose_landmarks.landmark
      # print('landmark')
      counter, status= TypeOfMove(landmarks).calculate_exercise(args["game_type"], counter, status)
      # print('final')
    except:
      pass
    score_table(args["game_type"], counter, status,0,timer(start_time))
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

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()