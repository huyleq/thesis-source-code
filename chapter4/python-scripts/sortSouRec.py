import numpy as np
from seppyio import *

ntrace=from_header('gygxsxsy','n2')
print 'ntrace',ntrace
gygxsxsy=read('gygxsxsy',(ntrace,4))

ind=np.zeros((ntrace,5))
ind[:,1:5]=gygxsxsy

for i in range(ntrace):
    ind[i,0]=i
ind=ind[ind[:,4].argsort()]
ind=ind[ind[:,3].argsort(kind='mergesort')]
ind=ind[ind[:,2].argsort(kind='mergesort')]
ind=ind[ind[:,1].argsort(kind='mergesort')]

ntrace=from_header('sorted_gyxsxy','n2')
ind=read('sorted_gyxsxy',(ntrace,5))
print 'ntrace',ntrace

recLineIndex=[0]
for i in range(1,ntrace):
    if ind[i,1]-ind[i-1,1]>300.:
        recLineIndex.append(i)
recLineIndex.append(ntrace)
nRecLine=len(recLineIndex)-1
print "there are",nRecLine,"receiver lines"

for i in range(nRecLine):
    b=recLineIndex[i]
    e=recLineIndex[i+1]
    print "receiver line",i,"has",e-b,"traces, starts at",b,"and ends at",e
    ind1=ind[ind[b:e,2].argsort(kind='mergesort')]
    ind[b:e,:]=ind1

write('out',ind)
to_header('out','n1',5,'o1',0,'d1',1)
to_header('out','n2',ntrace,'o2',0,'d2',1)
