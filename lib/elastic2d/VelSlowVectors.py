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

n=500
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

vectorp=np.zeros((n,4))
vectors=np.zeros((n,4))
vectorp1=np.zeros((n,4))
vectors1=np.zeros((n,4))

fig1=plt.figure(1)
ax1=fig1.add_subplot(111)
fig2=plt.figure(2)
ax2=fig2.add_subplot(111)
fig3=plt.figure(3)
ax3=fig3.add_subplot(111)
fig4=plt.figure(4)
ax4=fig4.add_subplot(111)

for i in range(n):
    kx=math.cos(theta[i])
    kz=math.sin(theta[i])
    Dkxdtheta=-kz
    Dkzdtheta=kx

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

    Dvpdtheta=xgp[i]*Dkxdtheta+ygp[i]*Dkzdtheta
    Dvsdtheta=xgs[i]*Dkxdtheta+ygs[i]*Dkzdtheta


    if(kz!=0):
        slope=-kx/kz
        slope2p1=slope*slope+1

        coeff=slope*xgp[i]-ygp[i]
        vectorp[i][2]=slope*coeff/slope2p1
        vectorp[i][3]=-coeff/slope2p1
        xi=np.linspace(-3,3,n)
        yi=slope*(xi-xgp[i])+ygp[i]
        ax1.plot(xi,yi,'k',lw=1)
        
        coeff=slope*xgs[i]-ygs[i]
        vectors[i][2]=slope*coeff/slope2p1
        vectors[i][3]=-coeff/slope2p1
        xi=np.linspace(-3,3,n)
        yi=slope*(xi-xgs[i])+ygs[i]
        ax2.plot(xi,yi,'k',lw=1)
    else:
        vectorp[i][2]=xgp[i]
        vectorp[i][3]=ygp[i]
        xi=np.ones(n)*xgp[i]
        yi=np.linspace(-3,3,n)
        ax1.plot(xi,yi,'k',lw=1)

        vectors[i][2]=xgs[i]
        vectors[i][3]=ygs[i]
        xi=np.ones(n)*xgs[i]
        yi=np.linspace(-3,3,n)
        ax2.plot(xi,yi,'k',lw=1)

    top=Dkzdtheta*vp-kz*Dvpdtheta
    bot=Dkxdtheta*vp-kx*Dvpdtheta
    if(bot!=0):
        slope=top/bot
        slope2p1=slope*slope+1
        coeff=slope*xsp[i]-ysp[i]
        vectorp1[i][2]=slope*coeff/slope2p1
        vectorp1[i][3]=-coeff/slope2p1
        xi=np.linspace(-3,3,n)
        yi=slope*(xi-xsp[i])+ysp[i]
        ax3.plot(xi,yi,'k',lw=1)
    else:
        vectorp1[i][2]=xsp[i]
        vectorp1[i][3]=ysp[i]
        xi=np.ones(n)*xsp[i]
        yi=np.linspace(-3,3,n)
        ax3.plot(xi,yi,'k',lw=1)

    top=Dkzdtheta*vs-kz*Dvsdtheta
    bot=Dkxdtheta*vs-kx*Dvsdtheta
    if(bot!=0):
        slope=top/bot
        slope2p1=slope*slope+1
        coeff=slope*xss[i]-yss[i]
        vectors1[i][2]=slope*coeff/slope2p1
        vectors1[i][3]=-coeff/slope2p1
        xi=np.linspace(-3,3,n)
        yi=slope*(xi-xss[i])+yss[i]
        ax4.plot(xi,yi,'k',lw=1)
    else:
        vectors1[i][2]=xss[i]
        vectors1[i][3]=yss[i]
        xi=np.ones(n)*xss[i]
        yi=np.linspace(-3,3,n)
        ax4.plot(xi,yi,'k',lw=1)

ax1.plot(xvp,yvp,'r-',lw=2.5,label='phase P-vel')
ax1.plot(xgp,ygp,'k-',lw=2.5,label='group P-vel')
X,Y,U,V=zip(*vectorp)
ax1.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,width=0.003)
ax1.set_xlabel(r'$k_x$',fontsize=15)
ax1.set_ylabel(r'$k_z$',fontsize=15)
ax1.set_xlim((-3,3))
ax1.set_ylim((-3,3))
ax1.legend()
temp=searchArgv('out1')
#plt.show()
#fig1.savefig(temp,bbox_inches='tight')


ax2.plot(xvs,yvs,'r-',lw=2.5,label='phase S-vel')
ax2.plot(xgs,ygs,'k-',lw=2.5,label='group S-vel')
X,Y,U,V=zip(*vectors)
ax2.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,width=0.003)
ax2.set_xlabel(r'$k_x$',fontsize=15)
ax2.set_ylabel(r'$k_z$',fontsize=15)
ax2.set_xlim((-3,3))
ax2.set_ylim((-3,3))
ax2.legend()
temp=searchArgv('out2')
#plt.show()
#fig2.savefig(temp,bbox_inches='tight')

ax3.plot(xsp,ysp,'k-',lw=2.5,label='phase P-slow')
ax3.plot(xgp1,ygp1,'r-',lw=2.5,label='group P-slow')
X,Y,U,V=zip(*vectorp1)
ax3.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,width=0.003)
ax3.set_xlabel(r'$k_x$',fontsize=15)
ax3.set_ylabel(r'$k_z$',fontsize=15)
ax3.set_xlim((-3,3))
ax3.set_ylim((-3,3))
ax3.legend()
temp=searchArgv('out3')
#plt.show()
fig3.savefig(temp,bbox_inches='tight')

ax4.plot(xgs1,ygs1,'r-',lw=2.5,label='group S-slow')
ax4.plot(xss,yss,'k-',lw=2.5,label='phase S-slow')
X,Y,U,V=zip(*vectors1)
ax4.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,width=0.003)
ax4.set_xlabel(r'$k_x$',fontsize=15)
ax4.set_ylabel(r'$k_z$',fontsize=15)
ax4.set_xlim((-3,3))
ax4.set_ylim((-3,3))
ax4.legend()
temp=searchArgv('out4')
#plt.show()
fig4.savefig(temp,bbox_inches='tight')



