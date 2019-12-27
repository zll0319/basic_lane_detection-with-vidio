# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:52:21 2019

@author: zll
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 

 
         
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): 
    lines =cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),\
                           minLineLength=min_line_len, maxLineGap=max_line_gap) 
    line_img =np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color=[ 0, 0,255]
    thickness=2
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_img,(x1,y1),(x2,y2),color,thickness)
    return line_img            

 
cap=cv2.VideoCapture("test_2.mp4")  
while (cap.isOpened()):
    ret,frame =cap.read()
    image = np.array(frame)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # 转为灰度图
    gaussian_blur_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)  # 高斯平滑，
    # 设置阈值，进行canny边缘提取
    canny_low_threshold = 30
    canny_high_threshold = 50
    edge_image = cv2.Canny(gaussian_blur_image, canny_low_threshold, canny_high_threshold)
    
    #plt.imshow(edge_image)
    #plt.show()
    # ROI可视区域选择，本程序选取左测公路区域
    image_shape = edge_image.shape
    '''
    x_offset = 200
    y_offset = 90
    
    v1 = (0,  image_shape[0] - y_offset*3)
    v2 = (int(image_shape[1]/4 + x_offset), int(image_shape[0]/3+y_offset))
    v3 = (int(image_shape[1]*5/8 - x_offset), int(image_shape[0]/3+y_offset))
    v4 = (image_shape[1] *2/ 3 + x_offset, image_shape[0] - y_offset )
    v5 = (0,  image_shape[0] - y_offset / 2 )
    
    vert = np.array([[v1, v2, v3,v4,v5]], dtype=np.int32)
    '''
    v1 = (250,  image_shape[0] )
    v2 = (1100, image_shape[0])
    v3 = (550, int(image_shape[0]*3/8))


#vert = np.array([[v1, v2, v3,v4,v5]], dtype=np.int32)
    vert = np.array([[v1, v2, v3]], dtype=np.int32)
    
    mask = np.zeros_like(edge_image)
    mask_color = 255
    # ROI可视区填充，在用mask与灰度图进行与运算，即在灰度图中得可视区
    
    cv2.fillPoly(mask, vert, mask_color)
    #plt.imshow( mask)
    #plt.show()
    
    masked_edge_image = cv2.bitwise_and(edge_image, mask)
    # 显示可视区的边缘提取二值图像
    #plt.imshow( masked_edge_image)
    #plt.show()
    #cv2.imshow("edge", masked_edge_image)
    #print(np.array([[v1, v2, v3, v4]], dtype=np.int32))
    # 霍夫线变换
    rho = 1         # 设置极径分辨率
    theta = (np.pi)/180  # 设置极角分辨率
    threshold = 70        # 设置检测一条直线所需最少的交点
    min_line_len =50    # 设置线段最小长度
    max_line_gap = 20   # 设置线段最近俩点的距离
    #lines = cv2.HoughLinesP(masked_edge_image, rho, theta, threshold, \
    #                        np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img =hough_lines(masked_edge_image, rho, theta, threshold, min_line_len, max_line_gap)
    hough_line_image = np.zeros((masked_edge_image.shape[0], masked_edge_image.shape[1], 3),\
                               dtype=np.uint8)
    # 绘制检测到的直线
    #plt.imshow(line_img)
    #plt.show()
    

    sync_vidio = cv2.addWeighted(frame, 0.8, line_img, 1,0)
    # 显示图像
    cv2.imshow("sync_vidio",sync_vidio)
    if cv2.waitKey(1) & 0xFF ==ord('q'): #Q键退出
        cap.release
        cv2.destroyAllWindows()
        break



