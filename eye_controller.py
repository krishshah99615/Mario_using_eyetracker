import cv2
import dlib
from math import hypot
import numpy as np
import os
from pynput.keyboard import Key, Controller
import time
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
keyboard = Controller()
BLINK_THRESH = 4.2
THRESHOLD_THRESH = 25

while(True):
    ret, frame = cap.read()
    flip = cv2.flip(frame, 1)
    keypoint_img = flip.copy()
    gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        ################## LANDMARKS ###############################
        for i in range(68):
            cv2.circle(keypoint_img, (landmarks.part(i).x,
                                      landmarks.part(i).y), 2, 255, -1)
        ################## LEFT EYE ###################################
        left_eye_left_pt = (landmarks.part(36).x, landmarks.part(36).y)
        left_eye_right_pt = (landmarks.part(39).x, landmarks.part(39).y)
        left_eye_top_mid = (int((landmarks.part(37).x+landmarks.part(38).x)/2),
                            int((landmarks.part(37).y+landmarks.part(38).y)/2))
        left_eye_bottom_mid = (int((landmarks.part(
            40).x+landmarks.part(41).x)/2), int((landmarks.part(40).y+landmarks.part(41).y)/2))

        left_hor_line = cv2.line(
            keypoint_img, left_eye_left_pt, left_eye_right_pt, (0, 255, 0), 1)
        left_ver_line = cv2.line(
            keypoint_img, left_eye_top_mid, left_eye_bottom_mid, (0, 255, 0), 1)

        left_hor_line_len = hypot(
            (left_eye_left_pt[0]-left_eye_right_pt[0]), (left_eye_left_pt[1]-left_eye_right_pt[1]))
        left_ver_line_len = hypot(
            (left_eye_top_mid[0]-left_eye_bottom_mid[0]), (left_eye_top_mid[1]-left_eye_bottom_mid[1]))

        left_open_eye_ratio = left_hor_line_len/left_ver_line_len

        left_eye_region = np.array([
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(37).x, landmarks.part(37).y),
            (landmarks.part(38).x, landmarks.part(38).y),
            (landmarks.part(39).x, landmarks.part(39).y),
            (landmarks.part(40).x, landmarks.part(40).y),
            (landmarks.part(41).x, landmarks.part(41).y),
        ], np.int32)

        left_min_x, left_min_y, left_max_x, left_max_y = np.min(left_eye_region[:, 0]), np.min(
            left_eye_region[:, 1]), np.max(left_eye_region[:, 0]), np.max(left_eye_region[:, 1])

        left_eye = cv2.resize(
            flip[left_min_y:left_max_y, left_min_x:left_max_x], None, fx=5, fy=5)
        left_gray_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        _, left_tresh_eye = cv2.threshold(
            left_gray_eye, THRESHOLD_THRESH, 255, cv2.THRESH_BINARY)
        le_h, le_w = left_tresh_eye.shape
        left_tresh_eye_half = left_tresh_eye[0:le_h, 0:int(le_w/2)]
        ################## RIGHT EYE ###################################
        right_eye_left_pt = (landmarks.part(42).x, landmarks.part(42).y)
        right_eye_right_pt = (landmarks.part(45).x, landmarks.part(45).y)
        right_eye_top_mid = (int((landmarks.part(43).x+landmarks.part(44).x)/2),
                             int((landmarks.part(43).y+landmarks.part(44).y)/2))
        right_eye_bottom_mid = (int((landmarks.part(47).x+landmarks.part(46).x)/2),
                                int((landmarks.part(47).y+landmarks.part(46).y)/2))
        right_hor_line = cv2.line(
            keypoint_img, right_eye_left_pt, right_eye_right_pt, (0, 255, 0), 1)
        right_ver_line = cv2.line(
            keypoint_img, right_eye_top_mid, right_eye_bottom_mid, (0, 255, 0), 1)

        right_hor_line_len = hypot(
            (right_eye_left_pt[0]-right_eye_right_pt[0]), (right_eye_left_pt[1]-right_eye_right_pt[1]))
        right_ver_line_len = hypot(
            (right_eye_top_mid[0]-right_eye_bottom_mid[0]), (right_eye_top_mid[1]-right_eye_bottom_mid[1]))

        right_open_eye_ratio = right_hor_line_len/right_ver_line_len

        right_eye_region = np.array([
            (landmarks.part(42).x, landmarks.part(42).y),
            (landmarks.part(43).x, landmarks.part(43).y),
            (landmarks.part(44).x, landmarks.part(44).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(46).x, landmarks.part(46).y),
            (landmarks.part(47).x, landmarks.part(47).y),
        ], np.int32)
        right_min_x, right_min_y, right_max_x, right_max_y = np.min(right_eye_region[:, 0]), np.min(
            right_eye_region[:, 1]), np.max(right_eye_region[:, 0]), np.max(right_eye_region[:, 1])

        right_eye = cv2.resize(
            flip[right_min_y:right_max_y, right_min_x:right_max_x], None, fx=5, fy=5)
        right_gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        _, right_tresh_eye = cv2.threshold(
            right_gray_eye, THRESHOLD_THRESH, 255, cv2.THRESH_BINARY)
        re_h, re_w = right_tresh_eye.shape
        right_tresh_eye_half = right_tresh_eye[0:re_h, int(re_w/2):re_w]
        ################# EYE INFO ###########################

        r = np.hstack((right_gray_eye, right_tresh_eye))
        l = np.hstack((left_gray_eye, left_tresh_eye))

        l_nz = cv2.countNonZero(left_tresh_eye_half)
        r_nz = cv2.countNonZero(right_tresh_eye_half)

        avg_open_eye_ratio = (right_open_eye_ratio+left_open_eye_ratio)/2

        ############## DISPLAY #####################
        os.system("cls")
        ctrl = ""
        if r_nz > l_nz:
            ctrl = "Right"
            keyboard.press(Key.right)

            print("CONTROL : "+ctrl)
        else:
            ctrl = "Left"
            keyboard.press(Key.left)

            print("CONTROL : "+ctrl)
        cv2.imshow("EYE INFO RIGHT", r)
        cv2.imshow("EYE INFO LEFT", l)

        print("AVG OPEN EYE RATIO : "+str(round(avg_open_eye_ratio, 3)))
        print("Right non zero : "+str(l_nz))
        print("left non zero : "+str(r_nz))
        if avg_open_eye_ratio > BLINK_THRESH:
            keyboard.press(Key.space)
            time.sleep(0. 5)
            keyboard.release(Key.space)
            print("STATUS : BLINKING")
        else:
            print("STATUS : NOT BLINKING")

    cv2.imshow("Facial Keypoints", keypoint_img)
    cv2.imshow("Orignal Image", flip)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
