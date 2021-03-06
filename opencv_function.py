#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:24:30 2019

@author: ishan
"""
# =============================================================================
# import sys,os
# try:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# except:
#     pass
# =============================================================================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from UndistortImage import UndistortImage
from ReadCameraModel import ReadCameraModel
import math
import glob



def getFundamentalMatrix(x1,x2):
    n = x1.shape[1]
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i],
        x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i],
        x2[0,i], x2[1,i], 1 ]
    
    
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    
 
    U,S,V = np.linalg.svd(F)
    
    S[2] = 0
    
    F = np.dot(U,np.dot(np.diag(S),V))
    return F

def get_R_T(F):
    
 
    E = np.dot(K.T,np.dot(F,K))
    
    U,S,V = np.linalg.svd(E) 
    
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))
  
    U,S,V = np.linalg.svd(E)
    
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
            
    t1 = U[:,2]  
    t2 = -U[:,2]          
    t3 = U[:,2]  
    t4 = -U[:,2]    
      
    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W,V))
    R3 = np.dot(U,np.dot(W.T,V))
    R4 = np.dot(U,np.dot(W.T,V))
# =============================================================================
#     R1 = U * W * V
#     R2 = U * W * V
#     R3 = U * W.T * V
#     R4 = U * W.T * V
# =============================================================================
    
    if np.linalg.det(R1)<0:
            R1 = -R1
            t1 = -t1
    elif np.linalg.det(R2)<0:
            R2 = -R2
            t2 = -t2
    elif np.linalg.det(R3)<0:
            R3 = -R3
            t3 = -t3  
    elif np.linalg.det(R4)<0:
            R4 = -R4
            t4 = -t4
    P = [np.vstack((R1,t1)).T,
         np.vstack((R2,t2)).T,
         np.vstack((R3,t3)).T,
         np.vstack((R4,t4)).T]
    return P


def triangulate(x1,x2,P1,P2):
    M = np.zeros((6,6))
    M[:3,:4] = P1  
    M[:2,4] = -x1
    M[2,4] = 1
    M[3:,:4] = P2
    M[3:5,5] = -x2
    M[5,5] = 1
    
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    return X / X[3]

def tri_pts(x1,x2,P1,P2):
    n = x1.shape[1]
    X = [triangulate(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return np.array(X).T

x_plot = []
z_plot = []
# =============================================================================
# 
# cap = cv2.VideoCapture('road.avi')
# while(True):
#     # Capture frame-by-frame
#     ret, frame1 = cap.read()
#     ret, frame2 = cap.read()
# =============================================================================
    

# =============================================================================
# path = Path(__file__).parent
# path /= "Oxford_dataset/stereo/centre/*.png"
# for f in path.iterdir():
#     print(f)    # <--- type: <class 'pathlib.PosixPath'>
#     f = str(f)  # <--- convert to string
#     img=cv2.imread(f)
# 
# =============================================================================
   
    
car = glob.glob("Oxford_dataset/stereo/centre/*.png")
car.sort()
car_images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for img in car]

print(len(car_images))

a = 0
N = np.zeros((4,4))
h = np.eye(4)
current_pos = np.zeros((3, 1))
current_rot = np.eye(3)
while a < (len(car_images) - 1):
    
# for i in range(len(car_images)):    
    #imageRaw1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    #rgb1 = cv2.cvtColor(imageRaw1, cv2.COLOR_BAYER_GR2BGR)
    rgb1 = cv2.cvtColor(car_images[a], cv2.COLOR_BAYER_GR2BGR)
 
    #imageRaw2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    #rgb2 = cv2.cvtColor(imageRaw2, cv2.COLOR_BAYER_GR2BGR)
    rgb2 = cv2.cvtColor(car_images[a + 1], cv2.COLOR_BAYER_GR2BGR)
    
    fx, fy, cx, cy, G_camera_image1, LUT1 = ReadCameraModel('Oxford_dataset/model')
    undistorted_image1 = UndistortImage(rgb1, LUT1)
    undistorted_image1 = cv2.cvtColor(undistorted_image1, cv2.COLOR_BGR2GRAY)
    
    eqimage1= cv2.equalizeHist(undistorted_image1)
    
    
    fx, fy, cx, cy, G_camera_image2, LUT2 = ReadCameraModel('Oxford_dataset/model')
    undistorted_image2 = UndistortImage(rgb2, LUT2)
    undistorted_image2 = cv2.cvtColor(undistorted_image2, cv2.COLOR_BGR2GRAY)
    
    eqimage2 = cv2.equalizeHist(undistorted_image2)
    
    
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]) 

    # Initiate STAR detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(eqimage1,None)
    kp2, des2 = orb.detectAndCompute(eqimage2,None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    good_points = []
    for match in matches:
        good_points.append(match)
        
    pts1 = np.float32([kp1[match.queryIdx].pt for match in good_points])
    pts2 = np.float32([kp2[match.trainIdx].pt for match in good_points])

    E,mask = cv2.findEssentialMat(pts2, pts1, K, cv2.FM_RANSAC, 0.999, 1.0, None)
    #pts1 = pts1[mask.ravel()==1]
    #pts2 = pts2[mask.ravel()==1]
    points, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    
# =============================================================================
#     s = 0.0
#     count = 0
#     for i in range(len(pts2) - 1):
#         for j in range(i+1,len(pts2)):
#             s = np.norm(pts2[i] - pts2[j]) / np.norm(pts1[i] - pts1[j])
#             
#             s += s
#             count += 1
#     s /= count        
#     
# =============================================================================
    
# =============================================================================
#     P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
#     P2 = get_R_T(F)
#     
#     indices = 0
#     max_res = 0
#     for i in range(4):
#         X = tri_pts(pts1.T,pts2.T,P1,P2[i])
#         d1 = np.dot(P1,X)[2]
#         d2 = np.dot(P2[i],X)[2]
#         if np.sum(d1>0) + np.sum(d2>0) > max_res:
#             max_res = np.sum(d1>0) + np.sum(d2>0)
#             indices = i
#             infront = (d1>0) & (d2>0)
#             
#     X = tri_pts(pts1.T,pts2.T,P1,P2[indices])        
#     #print(np.dot(P2[indices],X[:,infront])[2])
#     
#     
#     
#     P = P2[indices]
# =============================================================================
# =============================================================================
#     Rot = P[:3,:3]
#     print(Rot)
#     Trans = P[:3,3]
#     print(Trans)
#     current_pos += current_rot.dot(Trans) 
#     current_rot = np.dot(Rot,current_rot)
# =============================================================================
    
    current_pos += current_rot.dot(t) 
    current_rot = R.dot(current_rot)
    #print('cp',current_pos)
# =============================================================================
#     x1 = P[0,3]
#     z1 = P[2,3]
#     #print(P)
#     
#     N[:3,:] = P2[indices]
#     N[3,:] = [0,0,0,1]
#     
#     
#     h = np.dot(h,N)
#     x = h[0,3]
#     z = h[2,3]
# =============================================================================
    #print(h)
    
    x_plot.append(current_pos[0,0])
    z_plot.append(current_pos[2,0])
    
# =============================================================================
#     x_plot.append(-x)
#     z_plot.append(z)
# =============================================================================
    
    print(a)
    a+= 1

# =============================================================================
# x_plot = np.array(x_plot).reshape(x_plot.shape[0],-1)
# z_plot = np.array(z_plot).reshape(z_plot.shape[0],-1)
# final_array = np.vstack((x_plot,z_plot))
# 
# =============================================================================
final_array = []
for i in range(len(x_plot)):
    final_array.append((x_plot[i],z_plot[i]))
#print(final_array)
plt.scatter(x_plot,z_plot)
plt.show()


    
  
    
# =============================================================================
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
# 
# =============================================================================
