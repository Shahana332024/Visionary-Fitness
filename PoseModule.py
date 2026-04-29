import cv2
import mediapipe as mp
import math

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon > 0.5
        self.trackCon = trackCon > 0.5

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                      self.detectionCon, self.trackCon)

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def find_angle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.results.pose_landmarks.landmark[p1].x * img.shape[1], \
                 self.results.pose_landmarks.landmark[p1].y * img.shape[0]
        x2, y2 = self.results.pose_landmarks.landmark[p2].x * img.shape[1], \
                 self.results.pose_landmarks.landmark[p2].y * img.shape[0]
        x3, y3 = self.results.pose_landmarks.landmark[p3].x * img.shape[1], \
                 self.results.pose_landmarks.landmark[p3].y * img.shape[0]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw the angle lines
        if draw:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
            cv2.line(img, (int(x3), int(y3)), (int(x2), int(y2)), (255, 255, 255), 3)
            cv2.circle(img, (int(x1), int(y1)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x1), int(y1)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x2), int(y2)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x2), int(y2)), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(x3), int(y3)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x3), int(y3)), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (int(x2) - 50, int(y2) + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return angle
