import os
from shutil import copyfile
import random
import mediapipe as mp
import cv2
import numpy as np

def add_bg(images):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # Read images with OpenCV.
    
    # Show segmentation masks.
    # BG_COLOR = (192, 192, 192) # gray
    bg_list = os.listdir('bg/')
    this_bg = random.choice(bg_list)
    BG_COLOR = cv2.imread(f'bg/{this_bg}')
    with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
            # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
        results = selfie_segmentation.process(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(images.shape, dtype=np.uint8)
        # fg_image[:] = MASK_COLOR
        fg_image[:] = images[:]
        bg_image = np.zeros(images.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        # print(f'Segmentation mask of {name}:')
        return output_image
if __name__ == '__main__':
    print(os.listdir('bg/'))
    frame = cv2.imread('history_output/test1635312614.jpg_01260_00.jpg')
    cv2.imshow('show',frame)
    cv2.waitKey(0)
    frame = add_bg(frame)
    cv2.imshow('show',frame)
    cv2.waitKey(0)
