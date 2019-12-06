import math
import numpy as np

from seppyio import *

def H1(w,wi):
    return 0.5*(math.atan(w/wi)/wi-w/(w*w+wi*wi))
    
def H2(w,wi,wj):
    return (wi*math.atan(w/wi)-wj*math.atan(w/wj))/(wi*wi-wj*wj)
    
def H3(w,wi):
    return 0.5*math.log(w*w+wi*wi)

L=3

Qmin=get_param('Qmin')
Qmax=get_param('Qmax')
fmin=get_param('fmin')
fmax=get_param('fmax')

print 'Qmin=',Qmin,' Qmax=',Qmax
print 'fmin=',fmin,' fmax=',fmax

gbar=math.sqrt(Qmin*Qmax);
wmin=2*math.pi*fmin
wmax=2*math.pi*fmax
w=np.array([wmin,math.sqrt(wmin*wmax),wmax])

A=np.zeros((L,L))
b=np.zeros(L)
for i in range(L):
    b[i]=w[i]/gbar*(H3(wmax,w[i])-H3(wmin,w[i]))
    for j in range(L):
        if(i==j):
            A[i][i]=w[i]*w[i]*(H1(wmax,w[i])-H1(wmin,w[i]))
        else:
            A[i][j]=w[i]*w[j]*(H2(wmax,w[i],w[j])-H2(wmin,w[i],w[j]))

print 'A=',A
print 'b=',b

g=np.linalg.solve(A, b)
print 'gbar=',gbar
print 'g=',g
print 'omega=',w

