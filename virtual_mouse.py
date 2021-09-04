import numpy as np
import mediapipe as mp
import cv2 as cv
import time
import scipy.spatial as scipy_spatial
from collections import deque
import os
from sklearn.linear_model import LinearRegression
from pynput.mouse import Button, Controller
import subprocess


def get_main_point(pos1,pos2):
    distance=int(scipy_spatial.distance.euclidean(pos1,pos2)/2)
    return (pos2[0],pos2[1]+distance)

def creat_mask(points):
    mask=np.zeros((800,800))
    for x,y in points:
        mask[int(y)][x]=1
    return  mask

def Drow_vector(src,point1,point2):
    point3=(point1[0]+50,point1[1])
    cv.line(src,point1,point2,(255),2)
    cv.line(src,point1,point3,(100),2)

def drwa_line(src,points):
    for i in range(len(points)-1):
        cv.line(src,points[i],points[i+1],(255),2)

        
def cv_imshow(mask):
    cv.imshow('image',mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

def back_ground_substraction(shape,central_point,redius):
    h,w=shape
    motionMask=np.zeros((h,w),dtype=np.uint8)
    cv.circle(motionMask,central_point,redius,(255),cv.FILLED)
    return motionMask


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    #v1_u = unit_vector(v1)
    #v2_u = unit_vector(v2)
    v1=np.array(v1)
    v2=np.array(v2)
    dot=v1.dot(v2)
    det=v1[0]*v2[1]-v1[1]*v2[0]
    return math.degrees(math.atan2(det,dot))
    #return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def get_best_line(points):
    points=np.asarray(points)
    X=points[:,0]
    y=points[:,1]
    X_mean=np.mean(X)
    y_mean=np.mean(y)
    m=np.sum((X-X_mean)*(y-y_mean))/np.sum((X-X_mean)*(X-X_mean))
    b=y_mean-m*X_mean
    y=m*X+b
    return X,y 

def is_less_than_epsilon(point1,point2,epsilon):
    return abs(point1[0]-point2[0])+abs(point1[1]-point2[1])<epsilon

def get_resolution():
    cmd = ['xrandr']
    cmd2 = ['grep', '*']

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
    p.stdout.close()
    resolution_string, junk = p2.communicate()
    resolution = resolution_string.split()[0]
    width, height = resolution.decode('utf8').split('x')
    return int(width),int(height)


cap=cv.VideoCapture(0)
(screenx,screeny) =get_resolution()
(capturex,capturey) = (600,450)
cap.set(3,capturex)
cap.set(4,capturey)
mpHands=mp.solutions.hands
hands=mpHands.Hands(False,1,0.5,0.2)
mouse= Controller()
flag=False
time.sleep(2)
while True:
    # the time to check the frame rate processing
    start=time.time()
    
    _,frame=cap.read()
    h,w,c=frame.shape
    #y,x,c=frame.shape
    RGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result=hands.process(RGB)
    
    # if there is hand in the scene
    if result.multi_hand_landmarks:
        
        # take only one hand in the scene
        hand=result.multi_hand_landmarks[0]
        
        #the land mark for left click button and center position 
        left_click_point,left_click_point_2,center=hand.landmark[8],hand.landmark[4],hand.landmark[9]
        
        #the land mark for right click button
        right_click_point,right_click_point_2=hand.landmark[20],hand.landmark[4]
        
        # the land mark for left hold button
        hold_click_point,hold_click_point_2=hand.landmark[12],hand.landmark[4]
        
        # the position to check the distance for make left click action
        left_pos_1=(int(left_click_point.x*w),int(left_click_point.y*h))
        left_pos_2=(int(left_click_point_2.x*w),int(left_click_point_2.y*h))

        # the position to check the distance for make right click action
        right_pos_1=(int(right_click_point.x*w),int(right_click_point.y*h))
        right_pos_2=(int(right_click_point_2.x*w),int(right_click_point_2.y*h))

        # the position to check the distance for make hold click action
        hold_pos_1=(int(hold_click_point.x*w),int(hold_click_point.y*h))
        hold_pos_2=(int(hold_click_point_2.x*w),int(hold_click_point_2.y*h))

        
        cx,cy=(int(center.x*w),int(center.y*h))
        cv.circle(frame,(cx,cy),10,(0,255,0),cv.FILLED)
        current_position=mouse.position
        new_position=(screenx-(cx*screenx/capturex),screeny-(screeny-(cy*screeny/capturey)))
        if not is_less_than_epsilon(current_position,new_position,7):
            mouse.position=new_position
        
        # left distance
        left_distance=int(scipy_spatial.distance.euclidean(left_pos_1,left_pos_2))
        
#         #right distance
#         right_distance=int(scipy_spatial.distance.euclidean(right_pos_1,right_pos_2))
        
        #hold distance
        right_distance=int(scipy_spatial.distance.euclidean(hold_pos_1,hold_pos_2))

#         print(left_distance)
     
        if left_distance<=20:
            mouse.press(Button.left)
        else:
            mouse.release(Button.left)
        
        if right_distance<=20:
            mouse.press(Button.right)
            mouse.release(Button.right)
       

    end=1/(time.time()-start)
    cv.putText(frame,"FBS{:.2}".format(end),(15,15),cv.FONT_HERSHEY_SIMPLEX,0.5,(200,0,200),2)
    #cv.imshow('Mask',mask)
    cv.imshow('Frame',frame)
    #cv.imshow('Hsv',hsv)
    
    key=cv.waitKey(1)& 0XFF
    if key==ord('q'):
        cv.destroyAllWindows()
        break

cap.release()    
