import os
from shutil import copyfile
import subprocess
import time

import mediapipe as mp
import cv2
import numpy as np
# 濾鏡工具函式 
def modify_lightness_saturation(img, lightness=0, saturation=300):

    # 圖像歸一化，且轉換為浮點型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0
    
    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

#     lightness = 0 # lightness 調整為  "1 +/- 幾 %"
#     saturation = 300 # saturation 調整為 "1 +/- 幾 %"
 
    # 亮度調整
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
    
    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))

    return result_img

# 濾鏡工具函式 
def modify_color_temperature(img, cold_rate=20, hot_rate=0):
    
    # ---------------- 冷色調 ---------------- #  
    
#     height = img.shape[0]
#     width = img.shape[1]
#     dst = np.zeros(img.shape, img.dtype)

    # 1.計算三個通道的平均值，並依照平均值調整色調
    imgB = img[:, :, 0] 
    imgG = img[:, :, 1]
    imgR = img[:, :, 2] 

    # 調整色調請調整這邊~~ 
    # 白平衡 -> 三個值變化相同
    # 冷色調(增加b分量) -> 除了b之外都增加
    # 暖色調(增加r分量) -> 除了r之外都增加
    bAve = cv2.mean(imgB)[0] + hot_rate
    gAve = cv2.mean(imgG)[0] + cold_rate + hot_rate
    rAve = cv2.mean(imgR)[0] + cold_rate
    aveGray = (int)(bAve + gAve + rAve) / 3

    # 2. 計算各通道增益係數，並使用此係數計算結果
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve
    imgB = np.floor((imgB * bCoef))  # 向下取整
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    # 將原文第3部分的演算法做修改版，加快速度
    imgb = imgB
    imgb[imgb > 255] = 255
    
    imgg = imgG
    imgg[imgg > 255] = 255
    
    imgr = imgR
    imgr[imgr > 255] = 255
        
    cold_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8) 
            
    return cold_rgb

# 濾鏡工具函式 
def japanese_style_filter(img,output,file_name):
    # print("1. 調亮光線 (調整光線)")
    # print("2. 加強飽和度 (調整飽和度)")
    img = modify_lightness_saturation(img, lightness=0, saturation=50) # 單位: +- %
    
    # print("3. 將照片調成暖色調")
    img = modify_color_temperature(img, cold_rate=0, hot_rate=20)
    cv2.imwrite(f"{output}/{file_name}",img)




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
            japanese_style_filter(image,output,file_name)
            # cv2.imwrite(f"{output}/{file_name}", image)

#更換背景
def change_bg(path,outputs):
    IMAGE_FILES = [f"{path}/{file}"for file in os.listdir(path)]
    for img in IMAGE_FILES:
        output_name = img.split('/')[-1].split('.')[0]
        # run_win_cmd(f"backgroundremover -i {img} -a -ae 15 -o {outputs}/{output_name}.jpg")
        run_win_cmd(f"backgroundremover -i {img} -o {outputs}/{output_name}.jpg")
        front = cv2.imread(f'{outputs}/{output_name}.jpg',-1)
        # img2 = front[:,:,3]
        # background = cv2.imread('background.jpg',-1)
        background = np.ones((1024,768,3),dtype=np.uint8)*192
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
def copy_img(folder,test_folder):
    print('3')
    path = [f"{folder}\\{file}" for file in os.listdir(folder)]
    print('4')
    # print(path)
    for file in path:
        f = file.split('\\')[1]
        copyfile(file,f"VITON-HD/datasets/{test_folder}/image/{f}")
    print('5')


def test_pair(pair_folder,target_folder,txt_name,test_product):
    cloth = []
    with open(test_product,'r') as f:
        line = f.readline()
        while line:
            cloth.append(line[:-1])
            line = f.readline()

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
    print('1')
    change_bg(resized_folder,final_folder)
    print('2')
    copy_img(final_folder,test_folder)
    print('6')
    run_win_cmd(f'cd openpose && bin\\OpenPoseDemo.exe --image_dir ..\\{final_folder}\\ --display 0  --disable_blending --write_json ..\\VITON-HD\\datasets\\{test_folder}\\openpose-json --hand --write_images ..\\VITON-HD\\datasets\\{test_folder}\\openpose-img\\')
    print('7')
    run_win_cmd(f'cd human_parsing && python simple_extractor.py --dataset lip --model-restore checkpoints/final.pth --input-dir ..\\{resized_folder}\\ --output-dir ..\\VITON-HD\\datasets\\{test_folder}\\image-parse\\')
    test_pair('VITON-HD\\datasets',test_folder,test_pairs,test_product)

    run_win_cmd(f'cd VITON-HD && python test.py --name {output} --dataset_mode {test_folder} --dataset_list {test_pairs}')
    print(time.time()-start_time)