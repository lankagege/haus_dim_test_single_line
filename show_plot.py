#Dependecies: numpy, OpenCV, matplotlib, tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import multiprocessing
import pandas as pd

import os

a_list=[]
b_list=[]
for files in os.listdir('./save/'):
    csv_data_read=pd.read_csv('./save/'+files,header=None)
    flush=np.array(csv_data_read[[0,1]])
    a_list.append(np.array(flush[:,0]))
    b_list.append(np.array(flush[:,1]))

plt.scatter(a_list,b_list,s=1)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('a',fontsize=50)
plt.ylabel('b',fontsize=50)
plt.plot([0,1],[2,4.77258872],color='red')
plt.show()