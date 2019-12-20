#Dependecies: numpy, OpenCV, matplotlib, tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from tqdm import tqdm

#
ITERCOUNT=10000

#resultant a/b
ab_distrib=[]
ss=[]
cnts=[]
'''
for it in tqdm(range(ITERCOUNT)):
    ss+=[random.randrange(2,128)]
    cnts+=[random.randrange(1,256*128)]
    #regression
    ret = np.linalg.lstsq(np.vstack((np.log(np.divide(1,ss)),np.ones(len(ss)))).T,np.log(cnts),rcond=None)
    a,b=ret[0]
    ab_distrib+=[[a,b]]

#show result
nab_distrib=np.array(ab_distrib)
plt.scatter(nab_distrib[:,0],nab_distrib[:,1],s=1)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('a',fontsize=50)
plt.ylabel('b',fontsize=50)
plt.show()
'''

d=1

ss=[]
log_1_r=[]
for i in range(1,9+1):
    cnt=0
    s=256//d
    ss+=[s]
    d*=2
print(ss)
log_1_r=np.log(np.divide(1,ss))
print(log_1_r)
log_1_r_mean=np.mean(log_1_r ,dtype=np.float64)
print(log_1_r_mean)
print(np.subtract(log_1_r,log_1_r_mean))
print(sum(np.multiply(np.subtract(log_1_r,log_1_r_mean),np.subtract(log_1_r,log_1_r_mean))))


#
ITERCOUNT=10

#resultant a/b
ab_distrib=[]
line_length=[]
for it in tqdm(range(ITERCOUNT)):
    #init
    img_drawn=np.zeros((512,512),dtype=np.uint8)

    #draw a random single line
    #p1=(random.randrange(0,255),random.randrange(0,255))
    p1=(0,0)
    p2=(it,it)
    while p1==p2:
        p2=(random.randrange(0,512),random.randrange(0,512))
    img_drawn=cv2.line(img_drawn, p1, p2, 255)

    #box counting until single pixels
    d=2
    cnts=[]
    ss=[]
    for i in tqdm(range(2,256)):
        cnt=0
        s=512//d
        ss+=[s]
        for xi in range(d):
            for yi in range(d):
                roi=(xi*s, yi*s, s, s)
                cnt += 1 if np.count_nonzero(img_drawn[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]) else 0
        cnts+=[cnt]
        d=d+1

    #regression
    ret = np.linalg.lstsq(np.vstack((np.log10(np.divide(1,ss)),np.ones(len(ss)))).T,np.log10(cnts),rcond=None)
    a,b=ret[0]
    ab_distrib+=[[a,b]]
    line_length.append(it)

    #print('ab',a,b)
    #print('ri,ni',ss,cnts)

    #plt.scatter(np.log10(np.divide(1,ss)),np.log10(cnts))
    #np.log10(cnts)
    #plt.plot([min(np.log10(np.divide(1,ss))),max(np.log10(np.divide(1,ss)))],[a*min(np.log10(np.divide(1,ss)))+b,a*max(np.log10(np.divide(1,ss)))+b])
#plt.show()
#show result
nab_distrib=np.array(ab_distrib)
plt.subplot(121)
plt.scatter(nab_distrib[:,0],nab_distrib[:,1],s=3)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('a',fontsize=50)
plt.ylabel('b',fontsize=50)

plt.subplot(122)
plt.plot(line_length,nab_distrib[:,0],label='a')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('line_length',fontsize=50)

plt.plot(line_length,nab_distrib[:,1],label='b')
plt.legend(fontsize=20)
plt.show()




x=[]
y=[]
for key in gaus:
	x.append(key)
	y.append(gaus[key])

plt.scatter(x,y,s=4)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('length of line',fontsize=50)
plt.ylabel('number of samples',fontsize=50)
plt.show()
