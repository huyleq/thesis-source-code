#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

#include <errno.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <libssh/libssh.h>
#include <libssh/sftp.h>

#include "sshtunneling.h"

#include "myio.h"
#include "mylib.h"
#include "boundary.h"
#include "check.h"
#include "conversions.h"
#include "cluster.h"

using namespace std;

void objFuncGradientCij3d_network(float *fgcij,float *cij,int nx,int ny,int nz,float ox,float oy,float oz,float dx,float dy,float dz,string &cijfile,string &script,string &scriptpath,string &gradpath,string &outpath,string &datapath,string &command,int icall,ssh_session &session,sftp_session &sftp,float &time_in_min){
    chrono::high_resolution_clock::time_point start=chrono::high_resolution_clock::now();
    
    long long nxy=nx*ny,nxyz=nxy*nz,nn=3*nxyz;

    string cijheaderfile=gradpath+cijfile;
    string cijbinfile=datapath+cijfile+"@";

    cerr<<"writing binary file "<<cijbinfile<<endl;
    sftp_write((char *)cij,nn*sizeof(float),cijbinfile,session,sftp);
    
    string cijheader="n1="+to_string(nx)+" o1="+to_string(ox)+" d1="+to_string(dx)+"\n";
    cijheader+="n2="+to_string(ny)+" o2="+to_string(oy)+" d2="+to_string(dy)+"\n";
    cijheader+="n3="+to_string(nz)+" o3="+to_string(oz)+" d3="+to_string(dz)+"\n";
    cijheader+="n4=3\ndata_format=native_float\nesize=4\nin="+cijbinfile+"\n";
    cerr<<"writing header file "<<cijheaderfile<<endl;
    sftp_write((char *)cijheader.c_str(),cijheader.size(),cijheaderfile,session,sftp);
    
    string fgfile="fg_icall_"+to_string(icall)+".H";
    string fgheaderfile=gradpath+fgfile;
    string fgbinfile=datapath+fgfile+"@";
    string outfile=outpath+"run_icall_"+to_string(icall)+".log";
    string scriptfile=scriptpath+"run_icall_"+to_string(icall)+".sh";
    string script1=script+" icall="+to_string(icall)+" cij="+cijheaderfile+" fgcij="+fgheaderfile+" >& "+outfile;
    cerr<<"writing script file "<<scriptfile<<endl;
    sftp_write((char *)script1.c_str(),script1.size(),scriptfile,session,sftp);

    string output;
    string command1=command+" "+scriptfile;
    cerr<<"running command "<<command1<<endl;
    ssh_run_command(command1,session,output);
    cerr<<"output from running command "<<command1<<" is:"<<endl;
    cerr<<output<<endl;
    
    cerr<<"waiting for job on remote server for file "<<fgbinfile<<endl;
    string command2="ls -la "+fgbinfile;
    size_t nbyte=0;
    while(nbyte!=(nn+1)*sizeof(float)){
        this_thread::sleep_for(chrono::seconds(500));
        string output2;
        ssh_run_command(command2,session,output2);
        char *c=new char[output2.size()+1];
        char *c0=c;
        strcpy(c,output2.c_str());
        strtok(c," ");
        int count=0;
        while(c!=nullptr && count<5){
            nbyte=atoll(c);
            c=strtok(nullptr," ");
            count++;
        }
        delete []c0;
    }
    
    cerr<<"reading binary file "<<fgbinfile<<endl;
    sftp_read((char *)fgcij,(nn+1)*sizeof(float),fgbinfile,session,sftp);
    
    chrono::high_resolution_clock::time_point end=chrono::high_resolution_clock::now();
    chrono::duration<double> time=chrono::duration_cast<chrono::duration<double> >(end-start);
    time_in_min=time.count()/60.f;
    
    return;
}

