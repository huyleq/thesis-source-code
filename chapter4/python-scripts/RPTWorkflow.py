import numpy as np
from seppyio import *
import lasio

def overburden(wdepth,rho,dz):
    #s in psi, wdepth in ft, rho in g/cc, depth in ft
    #rho and depth from water bottom
    g=9.82;
    pair=14.7;
    pwater=0.000145038*g*1e3*wdepth*0.3048;
    n=rho.shape[0]
    s=np.zeros((n,1))
    s[0]=pair+pwater
    temp=0.000145038*0.5*1e3*0.3048*g*dz
    for i in range(1,n):
        s[i]=s[i-1]+temp*(rho[i]+rho[i-1])
    return s
    
def smectiteFraction(T,dt):
    #T in F and dt in My
    T1=(T+459.67)*5/9
    A=4e10
    E=20
    R=1.986e-3
    n=T.shape[0]
    smecFrac=np.zeros((n,1))
    integral=0.
    smecFrac[0]=1
    for i in range(1,n):
        kin=0.5*A*(np.exp(-E/R/T1[i])+np.exp(-E/R/T1[i-1])) 
        integral=integral+kin*dt
        smecFrac[i]=np.exp(-integral)
    
    return smecFrac

def betaFunction(smecFrac,beta0,beta1):
    return beta0*smecFrac+beta1*(1.-smecFrac)

def p2dtau(p,S,beta,dtaum,X,sigma0):
    sigma=S-p
    dtau=dtaum*np.power(1+1/beta[:,0]*np.log(sigma0/sigma),X)
    return dtau

def dtau2p(dtau,S,beta,dtaum,X,sigma0):
    # p,S,simga0 in psi,dtaum in mus/ft
    phig=np.power(dtau/dtaum,-1/X)
    phi=1-phig
    si=phi/phig
    sigma=sigma0*np.exp(-si*beta[:,0])
    p=S-sigma
    return p

def selectSandShale(depth,gamLog,sandPrctile,shalePrctile,log,depth0,depth1):
    sandDepth=[]
    sandGam=[]
    sand=[]
    shaleDepth=[]
    shaleGam=[]
    shale=[]
    gamLog1=gamLog[np.isfinite(gamLog)]
    gamSand=np.percentile(gamLog1,sandPrctile)
    gamShale=np.percentile(gamLog1,shalePrctile)
    dz=depth[1]-depth[0]
    n0=int((depth0-depth[0])/dz)
    n1=int((depth1-depth[0])/dz)
    for i in range(n0,n1):
        if np.isfinite(log[i]) and np.isfinite(gamLog[i]) and log[i]>0:
            if gamLog[i]<=gamSand:
                sandDepth.append(depth[i])
                sandGam.append(gamLog[i])
                sand.append(log[i])
            elif gamLog[i]>=gamShale:
                shaleDepth.append(depth[i])
                shaleGam.append(gamLog[i])
                shale.append(log[i])
    return sandDepth,sandGam,sand,shaleDepth,shaleGam,shale






