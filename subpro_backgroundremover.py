import os
from shutil import copyfile
import subprocess
import time

import mediapipe as mp
import cv2
import numpy as np


# 找出人物位置並置中 調成768*1024
def pretreat(path,output):
    mp_pose = mp.solutions.pose
    # For static images:
    IMAGE_FILES = [f"{path}/{file}"for file in os.listdir(path)]
    with mp_pose.Pose(
        static_image_mode=True,
        # model_complexity=2,
        # enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            file_name = file.split('/')[1]
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            all_land = np.zeros((33,3),dtype = np.int32)
            all_mark = results.pose_landmarks.landmark
            for i,v in enumerate(all_mark):
                all_land[i][0] = int(v.x * image_width)
                all_land[i][1] = int(v.y * image_height)
                all_land[i][2] = int(v.visibility * 100)
            #     image = cv2.circle(image,(all_land[i][0],all_land[i][1]),20,(100,100,100),-1)
            # image = cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

            human_width = all_land[:,0].max()-all_land[:,0].min()
            target_img_width = int(human_width * 2)
            target_img_origin_x = max(0,int(all_land[:,0].mean()-target_img_width/2))
            human_width = all_land[:,0].max()-all_land[:,0].min()
            target_img_width = int(human_width * 2.2)
            # target_img_origin_x = int(all_land[:,0].mean()-target_img_width/2)
            eye_lip_diff = (all_land[1,1]+all_land[4,1])/2-(all_land[9,1]+all_land[10,1])/2
            # print(all_land[25,2],all_land[26,2])
            if all_land[25,2]>60 or all_land[26,2]>60:
                buttom = int(min(all_land[25,1],all_land[26,1])-eye_lip_diff)
            elif (all_land[25,1]+all_land[26,1])-(all_land[11,1]+all_land[12,1])/2 < image_height:
                buttom = int((all_land[25,1]+all_land[26,1])-(all_land[11,1]+all_land[12,1])/2)
            else:
                buttom = image_height-10
            target_img_origin_y =max(0,int(2.85*(all_land[1,1]+all_land[4,1])/2-(all_land[9,1]+all_land[10,1])+eye_lip_diff))
            target_img_height = buttom-target_img_origin_y
            # print(target_img_origin_x,target_img_origin_y)
            # print(target_img_width,target_img_height)
            # print(f"{output}/{file_name}")
            image = image[target_img_origin_y:target_img_origin_y+target_img_height,target_img_origin_x:target_img_origin_x+target_img_width]
            image = cv2.resize(image,(768,1024))
            # cv2.imshow('test',image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(f"{output}/{file_name}", image)

#更換背景
def change_bg(path,outputs):
    IMAGE_FILES = [f"{path}/{file}"for file in os.listdir(path)]
    for img in IMAGE_FILES:
        output_name = img.split('/')[-1].split('.')[0]
        # run_win_cmd(f"backgroundremover -i {img} -a -ae 15 -o {outputs}/{output_name}.jpg")
        run_win_cmd(f"backgroundremover -i {img} -o {outputs}/{output_name}.jpg")
        front = cv2.imread(f'{outputs}/{output_name}.jpg',-1)
        img2 = front[:,:,3]
        background = cv2.imread('background.jpg',-1)
        conbine = np.zeros(background.shape,dtype=np.uint8)
        for i in range(background.shape[0]):
            for j in range(background.shape[1]):
                if front[i,j,3]>40:
                    conbine[i,j] = front[i,j,:-1]
                else:
                    conbine[i,j] = background[i,j,:]
        cv2.imwrite(f"{outputs}/{output_name}.jpg", conbine)



# 把final_imgs 放到測試資料夾裡
# 測試資料:'VITON-HD\\datasets\\test2\\image\\'
def copy_img(folder):
    path = [f"{folder}\\{file}" for file in os.listdir(folder)]
    # print(path)
    for file in path:
        f = file.split('\\')[1]
        copyfile(file,f"VITON-HD/datasets/test2/image/{f}")

#製造test_pair
def test_pair(pair_folder,target_folder,txt_name):
    cloth = os.listdir(f"{pair_folder}\{target_folder}\cloth")
    person = os.listdir(f"{pair_folder}\{target_folder}\image")
    with open(f"{pair_folder}\{txt_name}",'w') as f:
        for ip,p in enumerate(person):
            for ic,c in enumerate(cloth):
                if ip == len(person)-1 and ic == len(cloth)-1:
                    f.write(f"{p} {c}")
                else:
                    f.write(f"{p} {c}"+' \n')


#執行腳本來生出openpose的圖跟JSON以及Human Parsing
def run_win_cmd(cmd):
    result = []
    process = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    for line in process.stdout:
        result.append(line)
    errcode = process.returncode
    for line in result:
        print(line)
    if errcode is not None:
        raise Exception('cmd %s failed, see above for details', cmd)

if __name__ == '__main__':
    start_time = time.time()
    input_folder = 'tryimages'
    resized_folder = 'images_resize'
    final_folder = 'final_imgs'
    test_product = 'product.txt'
    test_folder = 'play'
    test_pairs = 'test_pairsplay.txt'
    output = 'play'

    pretreat(input_folder,resized_folder)
    change_bg(resized_folder,final_folder)
    copy_img(final_folder,test_folder)
    run_win_cmd(f'cd openpose && bin\\OpenPoseDemo.exe --image_dir ..\\{final_folder}\\ --display 0  --disable_blending --write_json ..\\VITON-HD\\datasets\\{test_folder}\\openpose-json --hand --write_images ..\\VITON-HD\\datasets\\{test_folder}\\openpose-img\\')
    run_win_cmd(f'cd human_parsing && python simple_extractor.py --dataset lip --model-restore checkpoints/final.pth --input-dir ..\\{resized_folder}\\ --output-dir ..\\VITON-HD\\datasets\\{test_folder}\\image-parse\\')
    test_pair('VITON-HD\\datasets',test_folder,test_pairs,test_product)

    run_win_cmd(f'cd VITON-HD && python test.py --name {output} --dataset_mode {test_folder} --dataset_list {test_pairs}')
    print(time.time()-start_time)