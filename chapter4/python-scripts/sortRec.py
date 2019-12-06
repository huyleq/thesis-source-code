import numpy as np
from seppyio import *

ntrace=from_header('gygxsxsy','n2')
print 'ntrace',ntrace
gygxsxsy=read('gygxsxsy',(ntrace,4))

ind=np.zeros((ntrace,5))
ind[:,1:5]=gygxsxsy

for i in range(ntrace):
    ind[i,0]=i
ind=ind[ind[:,2].argsort(kind='mergesort')]
ind=ind[ind[:,1].argsort(kind='mergesort')]

write('out',ind)
to_header('out','n1',5,'o1',0,'d1',1)
to_header('out','n2',ntrace,'o2',0,'d2',1)
