#ifndef SSHTUNNELING_H
#define SSHTUNNELING_H

#include <libssh/libssh.h>
#include <libssh/sftp.h>
#include <string>

//int verify_knownhost(ssh_session session);

//int authenticate_pubkey(ssh_session session);

int authenticate_password(ssh_session session);

int sftp_read(char *buffer,size_t nbyte,const std::string &filename,ssh_session &session,sftp_session &sftp);

int ssh_run_command(const std::string &command,ssh_session &session,std::string &output);

int sftp_write(char *buffer,size_t nbyte,const std::string &filename,ssh_session &session,sftp_session &sftp);

#endif
