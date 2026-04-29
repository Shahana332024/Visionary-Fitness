from flask import Flask, render_template, Response , send_from_directory
from flask_socketio import SocketIO , emit
import cv2
import numpy as np
from PoseModule import PoseDetector
import mediapipe as md
import math

app = Flask(__name__)


# Global variables to keep track of video capture and pose detector
cap = None
detector = PoseDetector()

pushups_count = 0
dumbbells_curl_count = 0
squats_count = 0
pullups_count = 0
N = 0






# somelist=["one","two","three"]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def get_image(filename):
    return send_from_directory('uploads',filename)


    

def track_pushups():
    global cap, pushups_count
    md_drawing = md.solutions.drawing_utils
    md_pose = md.solutions.pose
    position = None

    cap = cv2.VideoCapture(0)  # Initialize video capture
    with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Empty camera")
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            imlist = []

            if result.pose_landmarks:
                md_drawing.draw_landmarks(
                    image, result.pose_landmarks, md_pose.POSE_CONNECTIONS)
                for id, im in enumerate(result.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    X, Y = int(im.x * w), int(im.y * h)
                    imlist.append([id, X, Y])

            if len(imlist) != 0:
                if (imlist[12][2] >= imlist[14][2]) and (imlist[11][2] >= imlist[13][2]):
                    if (imlist[0][2] >= imlist[11][2]) and (imlist[0][2] >= imlist[12][2]):
                        position = "down"
                if ((imlist[12][2] and imlist[11][2]) <= (imlist[14][2] and imlist[13][2])) and position == "down":
                    position = "up"
                    pushups_count += 1
                    # print(count)
                    
                    
            # emit('pushups_count', pushups_count)
                    

                
            cv2.putText(image, str(int(pushups_count)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 4)


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release video capture when done

 # Adjust sleep interval as needed
        


@app.route('/video_feed_pushups')
def video_feed_pushups():
    return Response(track_pushups(), mimetype='multipart/x-mixed-replace; boundary=frame')

def track_dumbbell_curl():
    global cap, dumbbells_curl_count
    dir = 0


    cap = cv2.VideoCapture(0)  # Initialize video capture
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        img = detector.find_pose(img, draw=False)
        lmList = detector.find_position(img, draw=False)

        if len(lmList) != 0:
            # Right Arm
            angle_r = detector.find_angle(img, 12, 14, 16, True)
            # # Left Arm
            angle_l = detector.find_angle(img, 11, 13, 15, True)

            per_r = np.interp(angle_r, (200, 300), (0, 100))
            per_l = np.interp(angle_l, (200, 300), (0, 100))

            if per_r >= 100 and per_l >= 100:
                if dir == 0:
                    dir = 1
            if per_r <= 0 and per_l <= 0:
                if dir == 1:
                    dumbbells_curl_count += 1
                    dir = 0
            # print(count)
                    
            
        # emit('dumbbells_curl_count', dumbbells_curl_count)
        cv2.putText(img, str(int(dumbbells_curl_count)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 4)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release video capture when done

@app.route('/video_feed_dumbbell')
def video_feed_dumbbell():
    return Response(track_dumbbell_curl(), mimetype='multipart/x-mixed-replace; boundary=frame')



def track_squats():
    global cap,squats_count
    position = None
    md_drawing = md.solutions.drawing_utils
    md_pose = md.solutions.pose

    cap = cv2.VideoCapture(0)  # Initialize video capture
    with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Empty camera")
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            imlist = []

            if results.pose_landmarks:
                md_drawing.draw_landmarks(
                    image, results.pose_landmarks, md_pose.POSE_CONNECTIONS)

                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = image.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    imlist.append([id, x, y])

                if len(imlist) != 0:
                    left_hip = imlist[23][2]
                    right_hip = imlist[24][2]
                    left_knee = imlist[25][2]
                    right_knee = imlist[26][2]

                    if left_hip <= left_knee and right_hip <= right_knee:
                        position = "down"

                    if (left_hip > left_knee and right_hip > right_knee) and position == "down":
                        position = "up"
                        squats_count += 1

                #emit('squats_count',squats_count)
                cv2.putText(image, str(int(squats_count)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 4)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release video capture when done



@app.route('/video_feed_squats')
def video_feed_squats():
    return Response(track_squats(),mimetype='multipart/x-mixed-replace; boundary=frame')


def track_pullups():

    global cap, pullups_count

    md_drawing = md.solutions.drawing_utils
    md_pose = md.solutions.pose
    position = None
    #cap = cv2.VideoCapture('pull-ups - Made with Clipchamp.mp4')
    cap = cv2.VideoCapture(0)  # Initialize video capture
    with md_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Empty camera")
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            imlist = []

            if result.pose_landmarks:
                md_drawing.draw_landmarks(
                    image, result.pose_landmarks, md_pose.POSE_CONNECTIONS)
                for id, im in enumerate(result.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    X, Y = int(im.x * w), int(im.y * h)
                    imlist.append([id, X, Y])

            if len(imlist) != 0:
                if((imlist[12][2] and imlist[11][2]) <= (imlist[14][2] and imlist[13][2])) and position=="up":
                    position="down"
                    pullups_count +=1
                    print(pullups_count)  
                if(imlist[12][2] >= imlist[14][2]) and (imlist[11][2]>= imlist[13][2]):
                    position="up"
                    
                    
            # emit('pushups_count', pushups_count)
                    

                
            cv2.putText(image, str(int(pullups_count)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0), 4)


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  



@app.route('/video_feed_pullups')
def video_feed_pullups():
    return Response(track_pullups(),mimetype='multipart/x-mixed-replace; boundary=frame')



def track_jj():

    global cap,N
    # h = 0
    r = 0
    theta = 0
    setp = []
    points = []

    drawingModule = md.solutions.drawing_utils
    poseModule = md.solutions.pose
    pose = poseModule.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5)

    #cap = cv2.VideoCapture('jumpingJacks - Made with Clipchamp.mp4')
    cap = cv2.VideoCapture(0)
    cap.set(3, 1200)  # 3 for width
    cap.set(4, 900)  # 4 for height
    cap.set(10, 100)  # 10 for brightness

    # with poseModule: #(mode=False, upBody=False, smooth=True, detectionCon=0.5):
    while cap.isOpened():
        ret, frame = cap.read()
        
        #img2 = frame.copy()
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_height, image_width, _ = frame.shape

        if results.pose_landmarks:
            drawingModule.draw_landmarks(frame, results.pose_landmarks, poseModule.POSE_CONNECTIONS)
            for ids, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = lm.x * image_width, lm.y * image_height
                #print(ids, cx, cy)
                setp.append([cx, cy])

        if bool(setp):
            # h = setp[16][1] - setp[12][1]
            abx = setp[14][0] - setp[12][0]
            aby = setp[14][1] - setp[12][1]
            acx = setp[24][0] - setp[12][0]
            acy = setp[24][1] - setp[12][1]
            theta = math.acos((abx*acx + aby*acy)/((math.sqrt(abx**2 + aby**2))*(math.sqrt(acx**2 + acy**2))))
            #print(h)

        if theta < math.pi/4:
            theta1 = theta
        if theta > 3*math.pi/4:
            if theta1 < math.pi/4:
                N = N + 1
                theta1 = theta

        # if h > 0:
        #     h1 = h
        # if h < 0:
        #     if h1 > 0:
        #         N = N+1
        #         h1 = h

        # TO RESET THE COUNTER
        if bool(setp):
            r = math.sqrt((setp[12][0] - setp[19][0]) ** 2 + (setp[12][1] - setp[19][1]) ** 2)
        if r < 20:
            N = 0

        #print(N)
        
        cv2.putText(frame, str(int(N)), (40,100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 4)
        setp.clear()

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cv2.destroyAllWindows()
    cap.release()

@app.route('/video_feed_jj')
def video_feed_jj():
    return Response(track_jj(),mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/stopFeed')
def stopFeed():
    global cap

    cap.release()
    cv2.destroyAllWindows()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
