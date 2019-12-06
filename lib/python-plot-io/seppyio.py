import numpy as np
from numpy import array
import sys
import matplotlib.pyplot as plt
import os
from os.path import expanduser

def getFirstFromFile(filename,tag):
    found=False
    f=open(filename,'r')
    for line in f:
        for part in line.split(' '):
            j=part.find(tag+'=')
            if j!=-1:
                found=True
                val=part[j+len(tag)+1:]
                f.close()
                return val
    f.close()
    return found

def getAllFromFile(filename,tag,a):
    found=False
    f=open(filename,'r')
    for line in f:
        for part in line.split(' '):
            j=part.find(tag+'=')
            if j!=-1:
                found=True
                val=part[j+len(tag)+1:]
                a.append(val)
    f.close()
    return found

def searchFile(filename,tag):
    found=False
    f=open(filename,'r')
    for line in f:
        for part in line.split(' '):
            j=part.find(tag+'=')
            if j!=-1:
                found=True
                val=part[j+len(tag)+1:]
    f.close()
    if found==True:
        return val
    else: 
        print 'cannot find %s in %s\n' % (tag,filename)
        return found 

def searchArgv(tag):
    found=False
    for arg in sys.argv:
        i=arg.find(tag+'=')
        if i==0:
            val=arg[i+len(tag)+1:]
            found=True
    if found==True:
        return val
    else:
        print 'cannot find tag',tag
        return found

def get_sarray(parname):
    found=searchArgv(parname)
    if found!=False:
        return found.split(',')
    else:
        parfile=searchArgv('par')
        if parfile!=False:
            val=searchFile(parfile,parname)
            if val!=False:
                return val.split(',')
            else:
                print 'cannot find parameter %s\n' % parname
                return val
        else:
            print 'cannot find parameter file\n'
            return parfile

def get_array(parname):
    found=searchArgv(parname)
    if found!=False:
        if parname[0:1]=="n":
            return map(int,found.split(','))
        else:
            return map(float,found.split(','))
    else:
        parfile=searchArgv('par')
        if parfile!=False:
            val=searchFile(parfile,parname)
            if val!=False:
                if parname[0:1]=="n":
                    return map(int,val.split(','))
                else:
                    return map(float,val.split(','))
            else:
                print 'cannot find parameter %s\n' % parname
                return val
        else:
            print 'cannot find parameter file\n'
            return parfile

def get_param1(parname):
    found=searchArgv(parname)
    if found!=False:
        if parname[0:1]=="n":
            return int(found)
        else:
            return float(found)
    else:
        parfile=searchArgv('par')
        if parfile!=False:
            val=searchFile(parfile,parname)
            if val!=False:
                if parname[0:1]=="n":
                    return int(val)
                else:
                    return float(val)
            else:
                print 'cannot find parameter %s\n' % parname
                return val
        else:
            print 'cannot find parameter file for %s\n' % parname
            return parfile

def get_param(s1,s2=None,s3=None):
    a1=get_param1(s1)
    if s3!=None:
        a2=get_param1(s2)
        a3=get_param1(s3)
        return a1,a2,a3
    else:
        if s2!=None:
            a2=get_param1(s2)
            return a1,a2
        else:
            return a1

def to_header1(tag,parname,val):
    header=searchArgv(tag)
    if header==False:
        print 'cannot find header file for tag %s\n' % tag
        return header
    else:
        f=open(header,'a+')
        f.write(parname+'='+str(val)+'\n')
        f.close()
        return

def to_header(tag,par1,val1,par2=None,val2=None,par3=None,val3=None):
    to_header1(tag,par1,val1)
    if par2!=None:
        to_header1(tag,par2,val2)
    if par3!=None:
        to_header1(tag,par3,val3)
    return

def from_header1(tag,parname):
    header=searchArgv(tag)
    if header!=False:
        val=searchFile(header,parname)
        if val!=False:
            if parname[0:1]=="n":
                return int(val)
            else:
                return float(val)
        else:
            print 'cannot find parameter %s\n' % parname
            return val
    else:
        print 'cannot find header file for tag %s\n' % tag
        return header

def from_header(tag,par1,par2=None,par3=None):
    val1=from_header1(tag,par1)
    if par3!=None:
        val2=from_header1(tag,par2)
        val3=from_header1(tag,par3)
        return val1,val2,val3
    else: 
        if par2!=None:
            val2=from_header1(tag,par2)
            return val1,val2
        else:
            return val1

def read(tag,dim):
    header=searchArgv(tag)
    if header!=False:
        return sepread(header,dim)
    else:
        print 'cannot find header file for tag %s\n' % tag
        return header

def write(tag,buff):
    fullfilename=searchArgv(tag)
    sepwrite(fullfilename,buff)

def sepread(header,dim):
    binfile=searchFile(header,'in')
    if binfile!=False:
        binfile=binfile.rstrip('\n')
        binfile=binfile.strip('"')
        fm=searchFile(header,'data_format')
        if fm!=False:
            fm=fm.rstrip('\n')
            fm=fm.strip('"')
            if fm=='native_float':
                d=np.fromfile(binfile,dtype='=f4')
            elif fm=='xdr_float':
                d=np.fromfile(binfile,dtype='>f4')
            elif fm=='native_double':
                d=np.fromfile(binfile,dtype='=f8')
            elif fm=='xdr_double':
                d=np.fromfile(binfile,dtype='>f8')
            else:
                print 'format is none of permitted types: native_float, xdr_float, native_double, xdr_double'
            d=np.reshape(d,dim)
            return d
        else:
            print 'missing data_format'
            return fm
    else:
        print 'cannot find binary file'
        return binfile

def sepwrite(fullfilename,buff):
    datapathfile=expanduser('~')+'/.datapath'
    datapath=searchFile(datapathfile,'datapath')
    filename=os.path.basename(fullfilename)
    if filename!=False:
        binfile=datapath.rstrip('\n')+filename+'@'
        f=open(binfile,'wb+')
        buff=array(buff.flatten(),dtype='<f4')
        buff.tofile(f)
        f.close()
        f=open(fullfilename,'a+')
        for arg in sys.argv:
            f.write(arg+' ')
        f.write('\n')
        f.write('in='+binfile+'\n')
        f.write('data_format=native_float\n')
        f.close()
        return
    else:
        print 'cannot fine tag %s\n' % tag
        return

