import cv2
import mediapipe as mp
import numpy as np
import pickle
import time


def distance_between_2_points_2d(p1, p2):
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5


def draw_and_return_coords(img):
    coords = []
    hand_no = -1

    results = hands.process(img)
    if results.multi_hand_landmarks:
        no_of_hands = len(results.multi_hand_landmarks)
        for handLms in results.multi_hand_landmarks:
            hand_no += 1
            coords.append([])
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                coords[hand_no].append((cx, cy))
                # cv2.putText(img_black, str(id), (cx+10, cy), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 255), 2)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(img_black, handLms, mpHands.HAND_CONNECTIONS)

    return coords


# One Time Initialisations
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
gesture = ["close", "open", "yooo", "peace"]
ball_loc = (300, 300)
prev_time = 0

while True:
    img_black = np.zeros((480, 640, 3), np.uint8)
    Roi = []
    relative_coords = []
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_coords = draw_and_return_coords(imgRGB)

    if len(all_coords):  # if a hand is visibe
        xs = np.array([a[0] for a in all_coords[0]])
        ys = np.array([a[1] for a in all_coords[0]])
        min_x, min_y, max_x, max_y = np.min(
            xs), np.min(ys), np.max(xs), np.max(ys)
        Roi = img[min_y-10:max_y+10, min_x-10:max_x+10]
        for a in all_coords[0]:
            relative_coords.append(a[0]-min_x)
            relative_coords.append(a[1]-min_y)

        relative_coords = np.array(relative_coords).reshape(1, -1)
        pickle_in = open(__file__+"\\..\\gestures.pickle", 'rb')
        model = pickle.load(pickle_in)
        prediction = model.predict(relative_coords)
        confidence = model.predict_proba(relative_coords)

        if confidence[0][int(prediction)]*100 > 90:
            cv2.putText(img_black, gesture[int(
                prediction)]+" "+str(confidence[0][int(prediction)]*100), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.imshow("Roi", Roi)

            palm_centre = (int((all_coords[0][0][0]+all_coords[0][5][0]+all_coords[0][17][0])/3), int(
                (all_coords[0][0][1]+all_coords[0][5][1]+all_coords[0][17][1])/3))

            contidion1 = prediction == 0 and distance_between_2_points_2d(
                ball_loc, palm_centre) < 25
            condition2 = prediction == 0 and time.time()-prev_time < 0.5
            
            if contidion1 or condition2:
                ball_loc = palm_centre
                prev_time = time.time()

    cv2.circle(img, ball_loc, 15, (255, 255, 255), -1)
    cv2.circle(img_black, ball_loc, 15, (255, 255, 255), -1)

    cv2.imshow("Image", img)
    cv2.imshow("Image_black", img_black)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
