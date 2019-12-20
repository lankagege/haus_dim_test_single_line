#Dependecies: numpy, OpenCV, matplotlib, tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import math

x=[]
y=[]
for i in range(512):
    x.append(i)
    y.append(0.5*np.log10((i+i*i)/2))
x=np.array(x)
y=np.array(y)
plt.plot(x,y)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('x',fontsize=50)
plt.ylabel('y=0.5log((x+x^2)/2)',fontsize=50)
plt.show()