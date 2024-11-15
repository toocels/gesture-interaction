import cv2
import mediapipe as mp
import numpy as np
import pickle
import time


class ball:
    balls = []

    def __init__(self, colour_int, start):
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
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    def __init__(self):
        pass

    @staticmethod
    def hands_coords_calculator(img_temp):
        img_black = np.zeros((480, 640, 3), np.uint8)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
        coords = []
        hand_no = -1

        results = hand_skeleton.hands.process(img_temp)
        if results.multi_hand_landmarks:
            no_of_hands = len(results.multi_hand_landmarks)
            for handLms in results.multi_hand_landmarks:
                hand_no += 1
                coords.append([])
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm) #img
                    h, w, c = img_temp.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img_temp, (cx, cy), 15,
                               (255, 0, 255), cv2.FILLED)
                    coords[hand_no].append((cx, cy))
                    # cv2.putText(img_black, str(id), (cx+10, cy), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 255), 2)
                hand_skeleton.mpDraw.draw_landmarks(
                    img_black, handLms, hand_skeleton.mpHands.HAND_CONNECTIONS)

        return img_black, coords


class single_hand(hand_skeleton):
    def __init__(self, side):
        self.coords = []
        self.Roi = []
        self.side = side
        self.holding_ball = None
        self.prediction = None
        self.confidence = None
        self.palm_centre = None
        self.ball_history = [None, None, None, None, None,
                             None, None, None, None, None, None, None]

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

    def prediction_format(self, single_hand_coord):
        xs = np.array([a[0] for a in single_hand_coord])
        ys = np.array([a[1] for a in single_hand_coord])
        extremes = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

        return xs, ys, extremes

    def roi_cals(self):
        self.xs, self.ys, self.extremes = self.prediction_format(self.coords)
        self.relative_coords = gesture_control.relative_coords_cal(
            hand.coords, self.extremes)

    @staticmethod
    def left_or_right(hand_coords):
        if hand_coords[4][0] < hand_coords[17][0]:
            return 0
        else:
            return 1


class gesture_control:
    def __init__(self):
        self.gestures = []
        self.stat = 0
        self.sign_init_time = 0
        self.reset_time = 0

    @staticmethod
    def relative_coords_cal(hand_coords, extremes):
        relative_coords = []
        for a in hand_coords:
            relative_coords.append(a[0]-extremes[0])
            relative_coords.append(a[1]-extremes[1])
        relative_coords = np.array(
            relative_coords).reshape(1, -1)
        return relative_coords

    def check_reset(self):
        if self.stat == 1:
            if time.time()-self.sign_init_time > 2 and time.time()-reset.sign_init_time < 3:
                if self.gestures == [3, 3]:
                    print("reseted")
                    self.stat = 0
                    self.reset_time = time.time()
                    xs = [a for a in range(400, 230, -40)]
                    for i in range(0, 5):
                        ball.balls[i].current_pos = (xs[i], 240)
                else:
                    self.stat = 0
                    print("reset cancelled")

        elif self.gestures == [3, 3] and self.stat == 0 and time.time()-reset.reset_time > 3:  # not init
            if not time.time()-self.reset_time < 3:  # if time since reset >3
                print("sign initated")
                self.sign_init_time = time.time()
                self.stat = 1


# One Time Initialisations
cap = cv2.VideoCapture(0)
frame_rate_timer = 0

colours, xs = ["red", "green", "cyan", "gray",
               "white"], [a for a in range(400, 230, -40)]
for i in range(0, 5):
    ball_x = ball(colours[i], (xs[i], 240))

left_hand = single_hand(0)  # 0 measn L hand
right_hand = single_hand(1)  # 1 measn R hand
reset = gesture_control()

while True:
    frame_rate_timer = time.time()
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    img_drawn, all_hand_coords = hand_skeleton.hands_coords_calculator(img)
    visible_hands = []

    for i in all_hand_coords:  # Assign left and right hand
        if single_hand.left_or_right(i) == 0:
            left_hand.coords = i
            visible_hands.append(left_hand)
        elif single_hand.left_or_right(i) == 1:
            right_hand.coords = i
            visible_hands.append(right_hand)
        else:
            continue

    reset.gestures = []
    for hand in visible_hands:

        hand.roi_cals()
        hand.Roi = img[hand.extremes[1]-10:hand.extremes[3] +
                       10, hand.extremes[0]-10:hand.extremes[2]+10]

        pickle_in = open(__file__+"\\..\\gestures.pickle", 'rb')
        model = pickle.load(pickle_in)

        hand.prediction = model.predict(hand.relative_coords)[0]
        hand.confidence = model.predict_proba(hand.relative_coords)[
            0][hand.prediction]*100

        cv2.putText(img_drawn, str(hand.side)+" "+str(hand.prediction),
                    hand.coords[0], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        if hand.confidence > 80:
            hand.palm_centre = hand.centroid_calc(
                hand.coords[0], hand.coords[5], hand.coords[17])

            for i in ball.balls:
                condition1 = hand.prediction == 0 and hand.euclidian_dis(
                    i.current_pos, hand.palm_centre) < 35
                condition2 = hand.prediction == 0 and time.time(
                )-i.prev_time < 0.5 and i in hand.ball_history

                if (condition1 or condition2) and not hand.holding_ball:
                    hand.holding_ball = i
                    hand.ball_history.append(i)
                    hand.ball_history.pop(0)
                    i.current_pos = (hand.palm_centre[0], hand.palm_centre[1])
                    i.prev_time = time.time()
                    break
                else:
                    hand.holding_ball = 0

        reset.gestures.append(hand.prediction)

    reset.check_reset()

    for i in ball.balls:
        i.draw(img)
        i.draw(img_drawn)

    frame_rate = 1/(time.time()-frame_rate_timer)
    cv2.putText(img_drawn, str(int(frame_rate)),
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Image", cv2.flip(img, 1))
    cv2.imshow("Image_black", cv2.flip(img_drawn, 1))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
