

from flask import Flask, render_template, Response
import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from fitness.exercise import TypeOfExercise
from sounds.sound import fitness_sound
from interface.interface import TypeOfControl
from game.game import *
import pygame


app=Flask(__name__)

#ap = argparse.ArgumentParser()
#ap.add_argument("-t",
#                "--exercise_type",
#                type=str,
#                help='Type of activity to do',
#                required=True)
#ap.add_argument("-vs",
#                "--video_source",
#                type=str,
#                help='Type of activity to do',
#                required=False)
#args = vars(ap.parse_args())


## 強制進首頁使用control模式
args = {}
args['video_source'] = None
args["exercise_type"] = 'control'
args['type'] = 'fitness'


## drawing body
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def control_frames():  
    ## setting the video source
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(0)  # webcam
    w = 640
    h = 480
    cap.set(3, w)  # width
    cap.set(4, h)  # height

    with mp_pose.Pose(min_detection_confidence=0.8,
                  min_tracking_confidence=0.8) as pose:
        counter = 0  # movement of exercise
        status = True  # state of move
        hint = "Ready!"
        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)

            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            ## recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            ## make detection
            results = pose.process(frame)
            ## recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                counter, status, hint = TypeOfControl(landmarks).calculate_exercise(
                    args["exercise_type"], counter, status, hint)
            except:
                pass

            ## render detections (for landmarks)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255),
                                    thickness=2,
                                    circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45),
                                    thickness=2,
                                    circle_radius=2),
            )
            try:
                angle = BodyPartAngle(landmarks)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_left_arm())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_right_arm())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_KNEE'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_KNEE'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_left_leg())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (235, 150, 150), 2)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_right_leg())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (235, 150, 150), 2)
                cx = int(w *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_SHOULDER'].value].x+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_SHOULDER'].value].x)/2)
                cy = int(h *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_SHOULDER'].value].y+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_SHOULDER'].value].y)/2)
                cv2.putText(frame, str(round(angle.angle_of_the_neck())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 235, 150), 2)
                cx = int(w *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_HIP'].value].x+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].x)/2)
                cy = int(h *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_HIP'].value].y+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].y)/2)
                cv2.putText(frame, str(round(angle.angle_of_the_abdomen())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150), 2)
            except:
                pass

            #score_frame = score_table(args["exercise_type"], counter, status, hint)
            #print(frame.shape,score_frame.shape)
            #im_h_resize = cv2.hconcat([frame, score_frame])
            #ret, buffer = cv2.imencode('.jpg', im_h_resize)
            #frame = buffer.tobytes()
            frame = frame.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def fitness_frames(exercise_type):  
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(1)  # webcam
    w = 960
    h = 720
    cap.set(3, w)  # width
    cap.set(4, h)  # height

    ## 音效初始
    music = fitness_sound()

    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as pose:

        counter = 0  # movement of exercise
        status = True  # state of move
        hint = "Ready!"
        start_time = time.time()

        while cap.isOpened():
                    
            ret, frame = cap.read()
            if not ret:
                print("no ret")
            # result_screen = np.zeros((250, 400, 3), np.uint8)

            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            ## recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            ## make detection
            results = pose.process(frame)
            ## recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark            
                counter, status, hint= TypeOfExercise(landmarks).calculate_exercise(
                    exercise_type, counter, status, hint)            
                music.play_my_sound(exercise_type,int(timer(start_time)[-2:]))
                
                print(int(timer(start_time)[-2:]))
            except:            
                pass

            score_table(exercise_type, counter, status, hint, timer(start_time))
            
            ## render detections (for landmarks)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255),
                                    thickness=2,
                                    circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45),
                                    thickness=2,
                                    circle_radius=2),
            )
            try:
                angle = BodyPartAngle(landmarks)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_ELBOW'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_left_arm())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_ELBOW'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_right_arm())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 235), 2)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['LEFT_KNEE'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['LEFT_KNEE'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_left_leg())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (235, 150, 150), 2)
                cx = int(w *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].x)
                cy = int(h *landmarks[mp.solutions.pose.PoseLandmark['RIGHT_KNEE'].value].y)
                cv2.putText(frame, str(round(angle.angle_of_the_right_leg())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (235, 150, 150), 2)
                cx = int(w *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_SHOULDER'].value].x+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_SHOULDER'].value].x)/2)
                cy = int(h *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_SHOULDER'].value].y+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_SHOULDER'].value].y)/2)
                cv2.putText(frame, str(round(angle.angle_of_the_neck())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 235, 150), 2)
                cx = int(w *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_HIP'].value].x+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].x)/2)
                cy = int(h *(landmarks[mp.solutions.pose.PoseLandmark['LEFT_HIP'].value].y+landmarks[mp.solutions.pose.PoseLandmark['RIGHT_HIP'].value].y)/2)
                cv2.putText(frame, str(round(angle.angle_of_the_abdomen())), (cx-20, cy-20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (150, 150, 150), 2)
                
            except:
                pass

            # score_frame = score_table(exercise_type, counter, status, hint)
            # score_frame = cv2.resize(score_frame, (320,720), interpolation=cv2.INTER_AREA)
            # print(frame.shape,score_frame.shape)
            # im_h_resize = cv2.hconcat([frame, score_frame])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()            
# 文字資訊寫入txt檔
            with open('fitness.txt','w') as f:
                f.write(f"{exercise_type},{counter},{status},{hint}"+'\n')
# 網頁生成webcam影像
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def games_frames(game_type):
    if args["video_source"] is not None:
        cap = cv2.VideoCapture(args["video_source"])
    else:
        cap = cv2.VideoCapture(0)  # webcam
    w = 1600
    h = 1200
    cap.set(3, w)  # width
    cap.set(4, h)  # height   
    
    #音效初始
    file = f"sounds/game1.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    soundon = 0
    
    
    
    #設置遊戲初始環境
    start_time = time.time()
    env_list = game_start(game_type)
    counter = 0 # movement of exercise
    ## setup mediapipe
    with mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            # result_screen = np.zeros((250, 400, 3), np.uint8)
            frame = cv2.flip(frame,1)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            ## recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            ## make detection
            results = pose.process(frame)
            ## recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #================================================================

            #遊戲開始進行之後回傳給系統的參數
            env_coordinate = game_play(game_type,env_list,time.time()-start_time)
            #參數被畫在畫布上的樣子
            frame = game_plot(game_type,frame,env_coordinate)
            #================================================================
            try:
                if soundon==0 :
                    pygame.mixer.music.play()
                    soundon = 1
                    start_time = time.time()

                landmarks = results.pose_landmarks.landmark
                total_status = []
                for i,env in enumerate(env_coordinate):
                    counter, env_list[i].status = TypeOfMove(landmarks).calculate_exercise(game_type, counter, env[0],[w,h,env[1],env[2],env[3],env[4]])
                    total_status.append(env_list[i].status)

            except:
                total_status = []
                pass

            # score_table(game_type, counter, [str(x)[0] for x in total_status],timer(start_time))

            ## render detections (for landmarks)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255),
                                    thickness=2,
                                    circle_radius=2),
                mp_drawing.DrawingSpec(color=(174, 139, 45),
                                    thickness=2,
                                    circle_radius=2),
            )
            # score_frame = score_table(game_type, counter, env[0], [w,h,env[1],env[2],env[3],env[4]])
            # score_frame = cv2.resize(score_frame, (320,720), interpolation=cv2.INTER_AREA)
            # print(frame.shape,score_frame.shape)
            # im_h_resize = cv2.hconcat([frame, score_frame])
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # frame = frame.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        


## 主選單
@app.route('/')
def index():
    return render_template('index.html')

## cam測試用
@app.route('/video_feed')
def video_feed():
    return Response(control_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


## 運動頁面
@app.route('/fitness/<string:exercise_type>')
def fitness(exercise_type):
    return render_template('fitness.html',exercise_type=exercise_type)

## 遊戲頁面
@app.route('/game/<string:game_type>')
def game(game_type):
    return render_template('game.html',game_type=game_type)


## 運動頁面測試
@app.route('/fitness_feed/<string:exercise_type>')
def fitness_feed(exercise_type):
    return Response(fitness_frames(exercise_type), mimetype='multipart/x-mixed-replace; boundary=frame')

## 遊戲頁面測試
@app.route('/games_feed/<string:game_type>')
def games_feed(game_type):
    return Response(games_frames(game_type), mimetype='multipart/x-mixed-replace; boundary=frame')

## 健身選單
@app.route('/sport')
def sport():
    return render_template('sport.html')

#以下子豪新增部分
## 測試選單
@app.route('/test')
def test():
    return render_template('test.html',title = 'fitness_feed/squat')

# 傳字頁面
@app.route('/status_feed')
def status_feed():
    def generate():
        with open('fitness.txt','r') as f:
            yield f.read()  # return also will work
    return Response(generate(), mimetype='text') 
#以上子豪新增部分

if __name__=='__main__':
    import argparse
    app.run(debug=True)