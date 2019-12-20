#Dependecies: numpy, OpenCV, matplotlib, tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import multiprocessing
import pandas as pd
ITERCOUNT=[]
for i in range(10000):
    ITERCOUNT+=[i]


def cul_haus_dimention(ITERCOUNT):
    #resultant a/b
    #output_csv=open(str(ITERCOUNT)+'.csv','a')

    ab_distrib=[]

    for it in range(50000):
        #init
        img_drawn=np.zeros((256,256),dtype=np.uint8)

        #draw a random single line
        p1=(random.randrange(0,255),random.randrange(0,255))
        p2=p1
        while p1==p2:
            p2=(random.randrange(0,255),random.randrange(0,255))
        img_drawn=cv2.line(img_drawn, p1, p2, 255)

        #box counting until single pixels
        d=1
        cnts=[]
        ss=[]
        for i in range(1,9+1):
            cnt=0
            s=256//d
            ss+=[s]
            for xi in range(d):
                for yi in range(d):
                    roi=(xi*s, yi*s, s, s)
                    cnt += 1 if np.count_nonzero(img_drawn[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]) else 0
            cnts+=[cnt]
            d*=2

        #regression
        ret = np.linalg.lstsq(np.vstack((np.log(np.divide(1,ss)),np.ones(len(ss)))).T,np.log(cnts),rcond=None)
        a,b=ret[0]
        ab_distrib+=[[a,b]]
        #dataframe=pd.DataFrame()
        print('\r reading...'.format(it)+str(it),end='')
    np.savetxt('./save/'+str(ITERCOUNT)+'.csv', np.array(ab_distrib), delimiter = ',')
        #return np.array(ab_distrib)

#show result
pool=multiprocessing.Pool(processes=12)
for y in pool.imap(cul_haus_dimention,ITERCOUNT):
    print(y)
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
plt.show()