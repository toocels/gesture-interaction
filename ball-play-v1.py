from os import stat
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time


class ball:
    balls=[]

    def __init__(self, colour_int,start):
        self.colours = {"white": (255, 255, 255), "red": (0, 0, 255), "lime": (0, 255, 0), "blue": (255, 0, 0), "yellow": (
                        0, 255, 255), "cyan": (255, 255, 0), "gray": (128, 128, 128), "green": (0, 128, 0), "purple": (128, 0, 128)}
        self.colour = self.colours[colour_int]
        self.size = 15
        self.prev_pos = start
        self.current_pos = start
        self.prev_time = 0
        ball.balls.append(self)

    def draw(self, img):
        cv2.circle(img, self.current_pos, self.size, self.colour, -1)


class hand_skeleton:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def coords_calculator(self, img_temp):
        img_black = np.zeros((480, 640, 3), np.uint8)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
        coords = []
        hand_no = -1

        results = self.hands.process(img_temp)
        if results.multi_hand_landmarks:
            no_of_hands = len(results.multi_hand_landmarks)
            for handLms in results.multi_hand_landmarks:
                hand_no += 1
                coords.append([])
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm) #img
                    h, w, c = img_temp.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img_temp, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    coords[hand_no].append((cx, cy))
                    # cv2.putText(img_black, str(id), (cx+10, cy), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 255), 2)
                self.mpDraw.draw_landmarks(
                    img_black, handLms, self.mpHands.HAND_CONNECTIONS)

        return img_black, coords

    def euclidian_dis(self, p1, p2):
        return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5

    def centroid_calc(self, *coords):
        x_sum = 0
        y_sum = 0
        for i in coords:
            x_sum += i[0]
            y_sum += i[1]
        centroid = (int(x_sum/len(coords)), int(y_sum/len(coords)))
        return centroid

    def prediction_format(self, hand_coords):
        xs = np.array([a[0] for a in hand_coords])
        ys = np.array([a[1] for a in hand_coords])
        extremes = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

        return xs, ys, extremes


class gesture_control:
    status1=0
    status2=0

    def __init__(self):
        self.Roi = []
        self.relative_coords = []

    def relative_coords_cal(self,all_coords,extremes):
        for a in all_coords:
            gestures.relative_coords.append(a[0]-extremes[0])
            gestures.relative_coords.append(a[1]-extremes[1])
        gestures.relative_coords = np.array(gestures.relative_coords).reshape(1, -1)
        


# One Time Initialisations
cap = cv2.VideoCapture(0)
ball_1 = ball("red",(400, 240))
ball_2 = ball("green",(360, 240))
ball_3 = ball("cyan",(320, 240))
ball_4 = ball("gray",(280, 240))
ball_5 = ball("white",(240, 240))
skeleton = hand_skeleton()
gestures = gesture_control()

gesture = ["close", "open", "yooo", "peace"]

while True:
    
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    img_drawn, all_coords= skeleton.coords_calculator(img)

    for hand in all_coords:  # if a hand is visibe
        gestures.relative_coords = []
        xs, ys, extremes = skeleton.prediction_format(hand)
        gestures.relative_coords_cal(hand,extremes)

        gestures.Roi = img[extremes[1]-10:extremes[3]+10, extremes[0]-10:extremes[2]+10]
        
        pickle_in = open(__file__+"\\..\\gestures.pickle", 'rb')
        model = pickle.load(pickle_in)

        prediction, confidence = model.predict(gestures.relative_coords),model.predict_proba(gestures.relative_coords)[0]

        if confidence[int(prediction)]*100 > 80:
            cv2.putText(img_drawn, gesture[int(prediction)]+" "+str(confidence[int(
                prediction)]*100), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            palm_centre = skeleton.centroid_calc(
                hand[0], hand[5], hand[17])

            for i in ball.balls:
                condition1 = prediction == 0 and skeleton.euclidian_dis(
                i.current_pos, palm_centre) < 35
                condition2 = prediction == 0 and time.time()-i.prev_time < 0.5

                if (condition1 or condition2) and not gestures.status:
                    gestures.status=1
                    i.current_pos = (palm_centre[0], palm_centre[1])
                    i.prev_time = time.time()
                    break
                else:
                    gestures.status=0

    for i in ball.balls:
        i.draw(img)
        i.draw(img_drawn)

    cv2.imshow("Image", cv2.flip(img, 1))
    cv2.imshow("Image_black", cv2.flip(img_drawn, 1))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
