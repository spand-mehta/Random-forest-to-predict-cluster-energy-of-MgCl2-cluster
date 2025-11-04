# In[72]:
#19 MgCl2
import time
t=time.time()
import numpy as np
import pandas as pd
import sys

fpath=sys.argv[1]

df1=pd.read_csv(fpath,header=None)
import cPickle
with open('model.cpickle', 'rb') as f:
    regr = cPickle.load(f)

x=np.array(df1[0].T,dtype=float)
y=np.array(df1[1].T,dtype=float)

L=3.6363
xar=np.array(x[0:len(x)-2],dtype=float)
yar=np.array(y[0:len(x)-2],dtype=float)
xcar=np.array(x[0:len(x)-2],dtype=float)
ycar=np.array(y[0:len(x)-2],dtype=float)


for i,h in enumerate(xcar):
    xcar[i]=(h*(3.0**0.5)*3.6363/6)

for i in range(len(ycar)):
    ycar[i]=(ycar[i]+xar[i]/2)*(3.6363/3)

ycar

xcar,ycar

val=np.amin(xcar)
val1=2000.0
for j in range(len(xcar)):
    if(xcar[j]==val):
        if(ycar[j]<val1):
            val1=ycar[j]
            leftmost=j



def angle(a,b,c,d,e,f):

    ax=(a-c)*1.0
    ay=(b-d)*1.0;
    bx=(e-c)*1.0;
    by=(f-d)*1.0;

    #print a,b,c,d,e,f

    if((c==e) and (d==f)):
        #print "here:",c,d,e,f
        ang=-3
    elif((a==e) and (b==f)):
        ang=-3
    else:
        #print c,d,e,f
        costhetha=np.round(((ax*bx+ay*by)/(pow((ax*ax+ay*ay),0.5)*pow((bx*bx+by*by),0.5))),decimals=2);
        #print costhetha
        ang=np.degrees(np.arccos(costhetha));
    #print angles

    return ang


def findmax(angles,xcar,ycar,p):
    #print "angles",angles
    mx=np.where(angles==angles.max())
    #print "mx:",mx
    d=np.zeros(len(xcar))
    for i in range(len(mx)):
        d[mx]=(xcar[p]-xcar[mx])**2+(ycar[p]-ycar[mx])**2
        #print "d:",d
    p=np.where(d==d.max())

    return p


xsorted=np.zeros(len(xcar))
ysorted=np.zeros(len(ycar))
p=leftmost-1
c=0
f=0
while(p!=leftmost):
    f=f+1
    if(f==20):
        break
    if(c==0):
        p=leftmost
        xsorted[c]=xcar[p]
        ysorted[c]=ycar[p]
    #print "p:",c,p
    angles=np.arange(len(xcar))

    #print c,p,xsorted[c],ysorted[c]
    if(c==0):
        for i in range(len(xcar)):
            if(xcar[p]==0 and ycar[p]==0):
                angles[i]=angle(0,1,xcar[p],ycar[p],xcar[i],ycar[i])
            else:
                angles[i]=angle(xcar[p],ycar[p]+1,xcar[p],ycar[p],xcar[i],ycar[i])
        p=findmax(angles,xcar,ycar,p)
        #print "c:",c
        c=c+1
        xsorted[c]=xcar[p[0][0]]
        ysorted[c]=ycar[p[0][0]]
        #print c,xcar[p[0][0]],ycar[p[0][0]]
        #print "p:",c,p
        #print c,p[0][0],xsorted[c],ysorted[c]
    if(c!=0):
        for i in range(len(xcar)):
            angles[i]=angle(xsorted[c-1],ysorted[c-1],xsorted[c],ysorted[c],xcar[i],ycar[i])
            #print c,i,angles[i]
        p=findmax(angles,xcar,ycar,p)
        #print "c:",c
        c=c+1
        xsorted[c]=xcar[p[0][0]]
        ysorted[c]=ycar[p[0][0]]
        #print c,xcar[p[0][0]],ycar[p[0][0]]
        #print "p:",c,p
        if(p[0][0]==leftmost):
            break
        #print c,p[0][0],xsorted[c],ysorted[c]



xreqd=xsorted[0:c]
yreqd=ysorted[0:c]


var=(xcar-np.mean(xcar))**2+(ycar-np.mean(ycar))**2
radius_file=pow(np.sum(var/len(xcar)),0.5)


per=0;
for i in range(c):
    if(i!=c-1):
        per+=((xreqd[i]-xreqd[i+1])**2+(yreqd[i]-yreqd[i+1])**2)**0.5
    else:
        per+=((xreqd[i]-xreqd[0])**2+(yreqd[i]-yreqd[0])**2)**0.5
perimeter_file=per

are=0
for i in range(0,c):
    if(i!=c-1):
        a=(yreqd[i+1]-yreqd[i])*((xreqd[i+1]-xreqd[0])+(xreqd[i]-xreqd[0]))*0.5
        are+=a
    if(i==c-1):
        a=(yreqd[0]-yreqd[i])*((xreqd[i]-xreqd[0])+(xreqd[0]-xreqd[0]))*0.5
        are+=a
area_file=are
#print f1
compactness_file=(4*3.14*are/per**2)
#print "f12:",f1
per_div_ar_file=per/are

xdum=np.array(x[0:len(x)],dtype=float)
xwhole=np.arange(len(x),dtype=float)
ywhole=np.arange(len(y),dtype=float)
for i,h in enumerate(x):
    xwhole[i]=(h*(3.0**0.5)*3.6363/6)

for i in range(len(y)):
    ywhole[i]=(y[i]+xdum[i]/2)*(3.6363/3)

xwhole,ywhole

sym1_file=((np.mean(xwhole[-2:])-np.mean(xcar))**2+(np.mean(ywhole[-2:])-np.mean(ycar))**2)**0.5


d=0
for i in range(len(xcar)):
    for j in range(len(xcar)):
        dis=pow((pow((xcar[i]-xcar[j]),2)+pow((ycar[i]-ycar[j]),2)),0.5)
        if(dis>d):
            d=dis

sym2_file=(d-((xwhole[-2]-xwhole[-1])**2+(ywhole[-2]-ywhole[-1])**2)**0.5)/d

def nearneighbour(a,b,xcar,ycar,xn,yn):
    c=0
    for i in range(len(xcar)):
        dist=pow((pow((xcar[i]-a),2)+pow((ycar[i]-b),2)),0.5)
        if(abs(dist-3.6363)<0.01):
            #print a,b,xcar[i],ycar[i]
            xn[c]=xcar[i]-a;
            yn[c]=ycar[i]-b;
            #print c,a,b,xn[c],yn[c]
            c=c+1

    return c,xn,yn



con=np.zeros(len(xcar))
for j in range(len(xcar)):
    xv=0
    yv=0
    xn=np.zeros(len(xcar))
    yn=np.zeros(len(ycar))
    num,xn,yn=nearneighbour(xcar[j],ycar[j],xcar,ycar,xn,yn)
    if (num==5 or num==6):
        con[j]=6
    if (num==4):
        for i in range(num):
            xv+=xn[i]
            yv+=yn[i]
        #print j,xv,yv,xn,yn
        if(abs(pow((xv*xv+yv*yv),0.5)-6.29826)<0.01):
            con[j]=5
        else:
            con[j]=6
    if (num==3):
        for i in range(num):
            xv+=xn[i]
            yv+=yn[i]
        if((abs(xv)<0.01) and (abs(yv)<0.01)):
            con[j]=6
        elif(abs(pow((xv*xv+yv*yv),0.5)-3.6363)<0.01):
            con[j]=5
        else:
            con[j]=4
    if (num==2):
        for i in range(num):
            xv+=xn[i]
            yv+=yn[i]
        if(abs(pow((xv*xv+yv*yv),0.5)-6.29826)<0.01):
            con[j]=3
        else:
            con[j]=4
    if(num==1):
        con[j]=2

for i in range(len(xcar)):
    if((xwhole[-1]==xcar[i]) and (ywhole[-1]==ycar[i])):
        con[i]+=1
    if((xwhole[-2]==xcar[i]) and (ywhole[-2]==ycar[i])):
        con[i]+=1

cn2v=0
cn3v=0
cn4v=0
cn5v=0
cn6v=0
for i in range(len(xcar)):
    if(con[i]==2):
        cn2v+=1
    if(con[i]==3):
        cn3v+=1
    if(con[i]==4):
        cn4v+=1
    if(con[i]==5):
        cn5v+=1
    if(con[i]==6):
        cn6v+=1
cn2_file=cn2v
cn3_file=cn3v
cn4_file=cn4v
cn5_file=cn5v
cn6_file=cn6v
distance_file=((xwhole[-2]-xwhole[-1])**2+(ywhole[-2]-ywhole[-1])**2)**0.5

def findnearneighbour(a,b,c,d,xcar,ycar):
        nbdx=np.zeros(len(xcar))
        nbdy=np.zeros(len(ycar))
        nbdx1=np.zeros(len(xcar))
        nbdy1=np.zeros(len(ycar))

        c1=0
        d1=0

        for i in range(len(xcar)):
            dist=pow((pow((xcar[i]-a),2)+pow((ycar[i]-b),2)),0.5)
            #print a,b,dist
            if(abs(dist-3.6363)<0.01):
                nbdx[c1]=xcar[i]-a
                nbdy[c1]=ycar[i]-b
                c1=c1+1

        for i in range(len(xcar)):
            dist=pow((pow((xcar[i]-c),2)+pow((ycar[i]-d),2)),0.5)
            #print c,d,dist
            if(abs(dist-3.6363)<0.01):
                nbdx1[d1]=xcar[i]-c
                nbdy1[d1]=ycar[i]-d
                d1=d1+1

        xn1=np.sum(nbdx)*-1
        yn1=np.sum(nbdy)*-1
        xn2=np.sum(nbdx1)*-1
        yn2=np.sum(nbdy1)*-1
        #print nbdx,nbdy,nbdx1,nbdy1
        #print xn1,yn1,xn2,yn2

        if ((xn1**2+yn1**2)**0.5**0.5==0 or (xn2**2+yn2**2)**0.5==0):
            ang=-100
        else:
            cth=(xn1*xn2+yn1*yn2)/((xn1**2+yn1**2)*(xn2**2+yn2**2))**0.5
            ang=np.degrees(np.arccos(np.round(cth,decimals=2)))

        return ang

f=findnearneighbour(xwhole[-1],ywhole[-1],xwhole[-2],ywhole[-2],xcar,ycar)

anglet_file=f     

parameters=np.array([19,perimeter_file,distance_file,anglet_file,radius_file,sym1_file,sym2_file,per_div_ar_file,compactness_file,cn2_file,cn3_file,cn4_file,cn5_file,cn6_file])

print(regr.predict(parameters.reshape(1,-1)))[0]
#print time.time()-t