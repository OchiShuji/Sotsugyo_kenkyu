import cv2
import numpy as np
from matplotlib import pyplot as plt  
import csv

filename = "/Users/ochihideji/Desktop/a1_卒業研究/code/case14.jpg"
tgt_gry = cv2.imread(filename,0)
ret,tgt = cv2.threshold(tgt_gry,165,255,cv2.THRESH_BINARY)
tgt = cv2.bitwise_not(tgt)
cv2.imwrite('/Users/ochihideji/Desktop/a1_卒業研究/code/tgt.jpg', tgt)
contours,hierarchy = cv2.findContours(tgt,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
print(contours)
cnt2 = contours[1]
for k in range(1,len(contours)):
    cnt = contours[k]
    cntx = []
    cnty = []
    for i in range(0,len(cnt)):
        cntx.append(cnt[i][0][0])
        cnty.append(-cnt[i][0][1])
    plt.plot(cntx,cnty,color="black")
plt.show()
tgt_cnt = cv2.drawContours(tgt,contours, -1, (0,255,255), 2)
cv2.imwrite('/Users/ochihideji/Desktop/AA_航空/4s/a1_卒業研究/code/tgt_cnt.jpg', tgt_cnt)