#ifndef OBJFUNCGRADIENTCIJ3D_NETWORK_H
#define OBJFUNCGRADIENTCIJ3D_NETWORK_H

#include <string>
#include <libssh/libssh.h>
#include <libssh/sftp.h>

void objFuncGradientCij3d_network(float *fgcij,float *cij,int nx,int ny,int nz,float ox,float oy,float oz,float dx,float dy,float dz,std::string &cijfile,std::string &script,std::string &scriptpath,std::string &gradpath,std::string &outpath,std::string &datapath,std::string &command,int icall,ssh_session &session,sftp_session &sftp,float &time_in_minute);

#endif
