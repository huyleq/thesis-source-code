#plot velocity and slowness surfaces for VTI media from rho (g/cc), vp, vs (km/s), eps, del

import numpy as np
import math
import matplotlib.pyplot as plt

from seppyio import *

sqrt2=1.41421356237
twosqrt2=2.82842712475

rho,vp,vs=get_param("rho","vp","vs")
eps,delta=get_param("eps","del")

c33=rho*vp*vp
c44=rho*vs*vs
c11=c33*(1.+2.*eps)
c13=math.sqrt((c33-c44)*((1.+2.*delta)*c33-c44))-c44

#c11=39.4
#c13=5.8
#c33=13.1
#c44=3

C=np.zeros((3,3))
C[0][0]=c11
C[0][1]=c13
C[1][0]=c13
C[1][1]=c33
C[2][2]=c44

print "C=",C

n=1000
theta=np.linspace(0,2.*np.pi,n)
xvp=np.zeros(n)
yvp=np.zeros(n)
xvs=np.zeros(n)
yvs=np.zeros(n)
xsp=np.zeros(n)
ysp=np.zeros(n)
xss=np.zeros(n)
yss=np.zeros(n)
xgp=np.zeros(n)
ygp=np.zeros(n)
xgs=np.zeros(n)
ygs=np.zeros(n)
xgp1=np.zeros(n)
ygp1=np.zeros(n)
xgs1=np.zeros(n)
ygs1=np.zeros(n)
for i in range(n):
    kx=math.cos(theta[i])
    kz=math.sin(theta[i])

#    D=np.zeros((2,3))
#    D[0][0]=kx
#    D[0][2]=kz
#    D[1][1]=kz
#    D[1][2]=kx
#    A=np.matmul(D,np.matmul(C,np.transpose(D)))
#    eigs=np.linalg.eigvalsh(A)
#    vp=math.sqrt(np.amax(eigs)/rho)
#    vs=math.sqrt(np.amin(eigs)/rho)
    
    a=c11*kx*kx+c44*kz*kz
    b=(c13+c44)*kx*kz
    c=c44*kx*kx+c33*kz*kz
    amc=a-c
    apc=a+c
    d=math.sqrt(amc*amc+4.*b*b)
    apcpd=apc+d
    apcmd=apc-d
    sqrtapcpd=math.sqrt(apcpd)
    sqrtapcmd=math.sqrt(apcmd)
    vp=sqrtapcpd/sqrt2
    vs=sqrtapcmd/sqrt2
    
    sp=1./vp
    ss=1./vs
    xvp[i]=vp*kx
    yvp[i]=vp*kz
    xvs[i]=vs*kx
    yvs[i]=vs*kz
    xsp[i]=sp*kx
    ysp[i]=sp*kz
    xss[i]=ss*kx
    yss[i]=ss*kz

    Dvpda=(1.+amc/d)/sqrtapcpd/twosqrt2/math.sqrt(rho)
    Dvpdb=sqrt2*b/d/sqrtapcpd/math.sqrt(rho)
    Dvpdc=(1.-amc/d)/sqrtapcpd/twosqrt2/math.sqrt(rho)
    Dvsda=(1.-amc/d)/sqrtapcmd/twosqrt2/math.sqrt(rho)
    Dvsdb=-sqrt2*b/d/sqrtapcmd/math.sqrt(rho)
    Dvsdc=(1.+amc/d)/sqrtapcmd/twosqrt2/math.sqrt(rho)

    Dadkx=2.*c11*kx
    Dbdkx=(c13+c44)*kz
    Dcdkx=2.*c44*kx
    Dadkz=2.*c44*kz
    Dbdkz=(c13+c44)*kx
    Dcdkz=2.*c33*kz

    xgp[i]=Dvpda*Dadkx+Dvpdb*Dbdkx+Dvpdc*Dcdkx
    ygp[i]=Dvpda*Dadkz+Dvpdb*Dbdkz+Dvpdc*Dcdkz
    l=xgp[i]*xgp[i]+ygp[i]*ygp[i]
    xgp1[i]=xgp[i]/l
    ygp1[i]=ygp[i]/l
    xgs[i]=Dvsda*Dadkx+Dvsdb*Dbdkx+Dvsdc*Dcdkx
    ygs[i]=Dvsda*Dadkz+Dvsdb*Dbdkz+Dvsdc*Dcdkz
    l=xgs[i]*xgs[i]+ygs[i]*ygs[i]
    xgs1[i]=xgs[i]/l
    ygs1[i]=ygs[i]/l

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xvp,yvp,'r-',lw=2.5,label='phase P-vel')
ax.plot(xsp,ysp,'r--',lw=2.5,label='phase P-slow')
ax.plot(xgp,ygp,'k-',lw=2.5,label='group P-vel')
ax.plot(xgp1,ygp1,'k--',lw=2.5,label='group P-slow')
ax.set_xlabel(r'$k_x$',fontsize=15)
ax.set_ylabel(r'$k_z$',fontsize=15)
ax.set_xlim((-3,3))
ax.set_ylim((-3,3))
ax.legend()
temp=searchArgv('out1')
#plt.show()
plt.savefig(temp,bbox_inches='tight')

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xvs,yvs,'r-',lw=2.5,label='phase S-vel')
ax.plot(xss,yss,'r--',lw=2.5,label='phase S-slow')
ax.plot(xgs,ygs,'k-',lw=2.5,label='group S-vel')
ax.plot(xgs1,ygs1,'k--',lw=2.5,label='group S-slow')
ax.set_xlabel(r'$k_x$',fontsize=15)
ax.set_ylabel(r'$k_z$',fontsize=15)
ax.set_xlim((-3,3))
ax.set_ylim((-3,3))
ax.legend()
temp=searchArgv('out2')
#plt.show()
plt.savefig(temp,bbox_inches='tight')




