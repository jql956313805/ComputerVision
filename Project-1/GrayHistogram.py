# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 23:43:42 2022

@author: MSI
"""

from matplotlib import pyplot as plt
import cv2 
import numpy as np

def Draw_Histogram(hist,name):
    plt.figure()
    plt.title(name)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()

def Draw_RectText(img,x1,x2,y1,y2,text):
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),5)
    cv2.putText(img,text,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,4,(255,255,255),4)

first=cv2.imread('1st.jpg')
first=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)

dst11=first[607:923,1153:1557]
Draw_RectText(first,1153,1557,607,923,'1')

dst12=first[1677:2093,15:373]
Draw_RectText(first,15,373,1577,2093,'2')

dst13=first[3401:3751,1553:1967]
Draw_RectText(first,1553,1967,3401,3751,'3')

dst14=first[2163:2679,2683:3011]
Draw_RectText(first,2683,3011,2163,2679,'4')

second=cv2.imread('2nd.jpg')
second=cv2.cvtColor(second,cv2.COLOR_BGR2GRAY)

dst21=second[986:1308,2390:2694]
Draw_RectText(second,2390,2694,986,1308,'1')

dst22=second[950:1300,650:1000]
Draw_RectText(second,650,1000,950,1300,'2')

dst23=second[3316:3684,567:846]
Draw_RectText(second,567,846,3316,3684,'3')

dst24=second[3395:3793,2316:2620]
Draw_RectText(second,2316,2620,3395,3793,'4')

h11=cv2.calcHist([dst11],[0],None,[256],[0,256])
Draw_Histogram(h11,'Grayscale Histogram 1 - 1')
h11=cv2.normalize(h11, h11, 0, 1, cv2.NORM_MINMAX, -1)

h12=cv2.calcHist([dst12],[0],None,[256],[0,256])
Draw_Histogram(h12,'Grayscale Histogram 1 - 2')
h12=cv2.normalize(h12, h12, 0, 1, cv2.NORM_MINMAX, -1)

h13=cv2.calcHist([dst13],[0],None,[256],[0,256])
Draw_Histogram(h13,'Grayscale Histogram 1 - 3')
h13=cv2.normalize(h13, h13, 0, 1, cv2.NORM_MINMAX, -1)

h14=cv2.calcHist([dst14],[0],None,[256],[0,256])
Draw_Histogram(h14,'Grayscale Histogram 1 - 4')
h14=cv2.normalize(h14, h14, 0, 1, cv2.NORM_MINMAX, -1)

h21=cv2.calcHist([dst21],[0],None,[256],[0,256])
Draw_Histogram(h21,'Grayscale Histogram 2 - 1')
h21=cv2.normalize(h21, h21, 0, 1, cv2.NORM_MINMAX, -1)

h22=cv2.calcHist([dst22],[0],None,[256],[0,256])
Draw_Histogram(h22,'Grayscale Histogram 2 - 2')
h22=cv2.normalize(h22, h22, 0, 1, cv2.NORM_MINMAX, -1)

h23=cv2.calcHist([dst23],[0],None,[256],[0,256])
Draw_Histogram(h23,'Grayscale Histogram 2 - 3')
h23=cv2.normalize(h23, h23, 0, 1, cv2.NORM_MINMAX, -1)

h24=cv2.calcHist([dst24],[0],None,[256],[0,256])
Draw_Histogram(h24,'Grayscale Histogram 2 - 4')
h24=cv2.normalize(h24, h24, 0, 1, cv2.NORM_MINMAX, -1)

similarity11=cv2.compareHist(h11,h21,cv2.HISTCMP_CORREL)
similarity12=cv2.compareHist(h11,h22,cv2.HISTCMP_CORREL)
similarity13=cv2.compareHist(h11,h23,cv2.HISTCMP_CORREL)
similarity14=cv2.compareHist(h11,h24,cv2.HISTCMP_CORREL)
similarity1=np.array([similarity11,similarity12,similarity13,similarity14])
similarity1_max=max(similarity11,similarity12,similarity13,similarity14)

similarity21=cv2.compareHist(h12,h21,cv2.HISTCMP_CORREL)
similarity22=cv2.compareHist(h12,h22,cv2.HISTCMP_CORREL)
similarity23=cv2.compareHist(h12,h23,cv2.HISTCMP_CORREL)
similarity24=cv2.compareHist(h12,h24,cv2.HISTCMP_CORREL)
similarity2=np.array([similarity21,similarity22,similarity23,similarity24])
similarity2_max=max(similarity21,similarity22,similarity23,similarity24)

similarity31=cv2.compareHist(h13,h21,cv2.HISTCMP_CORREL)
similarity32=cv2.compareHist(h13,h22,cv2.HISTCMP_CORREL)
similarity33=cv2.compareHist(h13,h23,cv2.HISTCMP_CORREL)
similarity34=cv2.compareHist(h13,h24,cv2.HISTCMP_CORREL)
similarity3=np.array([similarity31,similarity32,similarity33,similarity34])
similarity3_max=max(similarity31,similarity32,similarity33,similarity34)

similarity41=cv2.compareHist(h14,h21,cv2.HISTCMP_CORREL)
similarity42=cv2.compareHist(h14,h22,cv2.HISTCMP_CORREL)
similarity43=cv2.compareHist(h14,h23,cv2.HISTCMP_CORREL)
similarity44=cv2.compareHist(h14,h24,cv2.HISTCMP_CORREL)
similarity4=np.array([similarity41,similarity42,similarity43,similarity44])
similarity4_max=max(similarity41,similarity42,similarity43,similarity44)

similarity_max=np.array([similarity1_max,similarity2_max,similarity3_max,similarity4_max])

second_point=np.array([[((2390+2694)//2)+3024,(986+1308)//2],[((650+1000)//2)+3024,(950+1300)//2],
                      [((567+846)//2)+3024,(3316+3684)//2],[((2316+2620)//2)+3024,(3395+3793)//2]])

result=cv2.hconcat([first,second])

for i in (similarity_max):
    if (i==similarity1_max):
        num=0
        for j in (similarity1):
            num+=1
            if(j==similarity1_max):
                cv2.line(result,((1153+1557)//2,(607+923)//2),(second_point[num-1][0],second_point[num-1][1]),(0,0,255),thickness=5,lineType=8)
    if (i==similarity2_max):
        num=0
        for j in (similarity2):
            num+=1
            if(j==similarity2_max):
                cv2.line(result,((15+373)//2,(1677+2093)//2),(second_point[num-1][0],second_point[num-1][1]),(0,0,255),thickness=5,lineType=8)
    if (i==similarity3_max):
        num=0
        for j in (similarity3):
            num+=1
            if(j==similarity3_max):
                cv2.line(result,((1553+1967)//2,(3401+3751)//2),(second_point[num-1][0],second_point[num-1][1]),(0,0,255),thickness=5,lineType=8)
    if (i==similarity4_max):
        num=0
        for j in (similarity4):
            num+=1
            if(j==similarity4_max):
                cv2.line(result,((2683+3011)//2,(2163+2679)//2),(second_point[num-1][0],second_point[num-1][1]),(0,0,255),thickness=5,lineType=8)

result=cv2.resize(result,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)
cv2.imshow('result',result)

print("similarity 1-4,2-2",similarity42)
print("similarity 1-4,2-4",similarity44)
"""
def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])    # X方向
    s_suanziY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])     
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
            new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5
    # return np.uint8(new_imageX)
    # return np.uint8(new_imageY)
    return np.uint8(new_image)    # 无方向算子处理的图像

def padding(image,ksize):# 输入目标图像，卷积核大小

    h = image.shape[0] # 获取图像尺寸
    w = image.shape[1]
    c = image.shape[2]

    pad = ksize // 2 # 需要补偿的边缘大小

    out_p = np.zeros((h+2*pad,w+2*pad,c)) # 创造一个补偿后大小的全0矩阵

    out_copy = image.copy()

    out_p[pad:pad+h,pad:pad+w,0:c] = out_copy.astype(np.uint8) # 将原图像复制入目标图像中
    
    return out_p

def gaussian(image,ksize):
    sigma =  0.3 *((ksize-1)*0.5-1) + 0.8
    pad = ksize//2

    out_p = padding(image,ksize) # padding之后的图像
    # print(out_p)

    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]

    # 高斯卷积核

    kernel = np.zeros((ksize,ksize))
    for x in range(-pad,-pad+ksize):
        for y in range(-pad,-pad+ksize):
            kernel[y+pad,x+pad] = np.exp(-(x**2+y**2)/(2*(sigma**2)))
    kernel /= (sigma*np.sqrt(2*np.pi))
    kernel /=  kernel.sum()

    # print(kernel)

    tmp = out_p.copy()

    # print(tmp)

    for y in range(h):
        for x in range(w):
            for z in range(c):

                out_p[pad+y,pad+x,z] = np.sum(kernel*tmp[y:y+ksize,x:x+ksize,z])


    out = out_p[pad:pad+h,pad:pad+w].astype(np.uint8)
    # print(out)

    return out

gradient11=gaussian(dst11,3)
sobel_suanzi(dst11)

gradient12=gaussian(dst12,3)
sobel_suanzi(dst12)

gradient13=gaussian(dst13,3)
sobel_suanzi(dst13)

gradient14=gaussian(dst14,3)
sobel_suanzi(dst14)

gradient21=gaussian(dst21,3)
sobel_suanzi(dst21)

gradient22=gaussian(dst22,3)
sobel_suanzi(dst22)

gradient23=gaussian(dst23,3)
sobel_suanzi(dst23)

gradient24=gaussian(dst24,3)
sobel_suanzi(dst24)

gradient11_hist=plt.hist(dst11, bins = 5) 
plt.title('Gradient Histogram 1 - 1')
plt.show(gradient11_hist)
gradient12_hist=plt.hist(dst12, bins = 5) 
plt.title('Gradient Histogram 1 - 2')
plt.show(gradient12_hist)
gradient13_hist=plt.hist(dst13, bins = 5) 
plt.title('Gradient Histogram 1 - 3')
plt.show(gradient13_hist)
gradient14_hist=plt.hist(dst14, bins = 5) 
plt.title('Gradient Histogram 1 - 4')
plt.show(gradient14_hist)
gradient21_hist=plt.hist(dst21, bins = 5) 
plt.title('Gradient Histogram 2 - 1')
plt.show(gradient21_hist)
gradient22_hist=plt.hist(dst22, bins = 5) 
plt.title('Gradient Histogram 2 - 2')
plt.show(gradient22_hist)
gradient23_hist=plt.hist(dst23, bins = 5) 
plt.title('Gradient Histogram 2 - 3')
plt.show(gradient23_hist)
gradient24_hist=plt.hist(dst24, bins = 5) 
plt.title('Gradient Histogram 2 - 4')
plt.show(gradient24_hist)
"""
cv2.waitKey()
cv2.destroyAllWindows()