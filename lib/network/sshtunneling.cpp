#define LIBSSH_STATIC 1
#include <cmath>
#include <errno.h>
#include <string.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <libssh/libssh.h>
#include <libssh/sftp.h>

#include "sshtunneling.h"

#define BUF_SIZE 16384

using namespace std;

//int verify_knownhost(ssh_session session) {
//    unsigned char *hash = NULL;
//    ssh_key srv_pubkey = NULL;
//    size_t hlen;
//    char buf[10];
//    int rc = ssh_get_server_publickey(session, &srv_pubkey);
//    if (rc < 0) return -1;
//    rc = ssh_get_publickey_hash(srv_pubkey,SSH_PUBLICKEY_HASH_SHA1,&hash,&hlen);
//    ssh_key_free(srv_pubkey);
//    if (rc < 0) return -1;
//    enum ssh_known_hosts_e state = ssh_session_is_known_server(session);
//    switch (state) {
//        case SSH_KNOWN_HOSTS_OK:
//            /* OK */
//            break;
//        case SSH_KNOWN_HOSTS_CHANGED:{
//            fprintf(stderr, "Host key for server changed: it is now:\n");
//            ssh_print_hexa("Public key hash", hash, hlen);
//            fprintf(stderr, "For security reasons, connection will be stopped\n");
//            ssh_clean_pubkey_hash(&hash);
//            return -1;
//        }
//        case SSH_KNOWN_HOSTS_OTHER:{
//            fprintf(stderr, "The host key for this server was not found but an other"
//                    "type of key exists.\n");
//            fprintf(stderr, "An attacker might change the default server key to"
//                    "confuse your client into thinking the key does not exist\n");
//            ssh_clean_pubkey_hash(&hash);
//            return -1;
//        }
//        case SSH_KNOWN_HOSTS_NOT_FOUND:{
//            fprintf(stderr, "Could not find known host file.\n");
//            fprintf(stderr, "If you accept the host key here, the file will be"
//                    "automatically created.\n");
//        }
//            /* FALL THROUGH to SSH_SERVER_NOT_KNOWN behavior */
//        case SSH_KNOWN_HOSTS_UNKNOWN:{
//            char *hexa = ssh_get_hexa(hash, hlen);
//            fprintf(stderr,"The server is unknown. Do you trust the host key?\n");
//            fprintf(stderr, "Public key hash: %s\n", hexa);
//            ssh_string_free_char(hexa);
//            ssh_clean_pubkey_hash(&hash);
//            char *p = fgets(buf, sizeof(buf), stdin);
//            if (p == NULL) return -1;
//            int cmp = strncasecmp(buf, "yes", 3);
//            if (cmp != 0) return -1;
//            rc = ssh_session_update_known_hosts(session);
//            if (rc < 0) {
//                fprintf(stderr, "Error %s\n", strerror(errno));
//                return -1;
//            }
//            break;
//        }
//        case SSH_KNOWN_HOSTS_ERROR:{
//            fprintf(stderr, "Error %s", ssh_get_error(session));
//            ssh_clean_pubkey_hash(&hash);
//            return -1;
//        }
//    }
//    ssh_clean_pubkey_hash(&hash);
//    return 0;
//}
//
//int authenticate_pubkey(ssh_session session) {
//  int rc = ssh_userauth_publickey_auto(session, NULL, NULL);
//  if (rc == SSH_AUTH_ERROR) {
//     fprintf(stderr, "Authentication failed: %s\n",ssh_get_error(session));
//     return SSH_AUTH_ERROR;
//  }
//  return rc;
//}

int authenticate_password(ssh_session session) {
  char *password = getpass("Enter your password: ");
  int rc = ssh_userauth_password(session, NULL, password);
  if (rc == SSH_AUTH_ERROR) {
     fprintf(stderr, "Authentication failed: %s\n",ssh_get_error(session));
     return SSH_AUTH_ERROR;
  }
  return rc;
}

int sftp_read(char *buffer,size_t nbyte,const string &filename,ssh_session &session,sftp_session &sftp){
  int access_type = O_RDONLY;;
  sftp_file file = sftp_open(sftp, filename.c_str(),access_type, 1);
  if (file == NULL) {
      fprintf(stderr, "Can't open file %s for reading. Error: %s\n",filename.c_str(),ssh_get_error(session));
      return SSH_ERROR;
  }
//  int nbyteread = sftp_read(file, buffer, nbyte);
  size_t counter=0;
  size_t nleft=nbyte;
  int nread;
  while(counter<nbyte){
      size_t n=min(nleft,(size_t)BUF_SIZE);
      nread=sftp_read(file,buffer+counter,n);
//      fprintf(stderr,"n=%d nwritten%d nleft=%d counter=%d\n",n,nwritten,nleft,counter);
      counter+=n;
      nleft-=n;
  }
  if (counter!=nbyte || nleft>0) {
    fprintf(stderr, "Error reading data from file %s.Error: %s\n",filename.c_str(),ssh_get_error(session));
    sftp_close(file);
    return SSH_ERROR;
  }
  int rc = sftp_close(file);
  if (rc != SSH_OK) {
      fprintf(stderr, "Can't close the read file %s. Error: %s\n",filename.c_str(),ssh_get_error(session));
      return rc;
  }
  return SSH_OK;
}

int ssh_run_command(const string &command,ssh_session &session,string &output){
  ssh_channel channel = ssh_channel_new(session);
  char buffer[256];
  if (channel == NULL){
    fprintf(stderr,"Can't create channel to run command %s\n",command.c_str());
    return SSH_ERROR;
  }
  int rc = ssh_channel_open_session(channel);
  if (rc != SSH_OK){
    fprintf(stderr,"Can't open channel to run command %s\n",command.c_str());
    ssh_channel_free(channel);
    return rc;
  }
  rc = ssh_channel_request_exec(channel, command.c_str());
  if (rc != SSH_OK){
    fprintf(stderr,"Can't run command %s\n",command.c_str());
    ssh_channel_close(channel);
    ssh_channel_free(channel);
    return rc;
  }
  int nbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
  while (nbytes > 0){
    char *buffer1=new char[nbytes];
    memcpy(buffer1,buffer,nbytes);
    output+=buffer1;
    delete []buffer1;
    nbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
  }
  if (nbytes < 0){
    ssh_channel_close(channel);
    ssh_channel_free(channel);
    return SSH_ERROR;
  }
  ssh_channel_send_eof(channel);
  ssh_channel_close(channel);
  ssh_channel_free(channel);
  return SSH_OK;
}

int sftp_write(char *buffer,size_t nbyte,const string &filename,ssh_session &session,sftp_session &sftp) {
  int access_type = O_WRONLY | O_CREAT | O_TRUNC;
  sftp_file file = sftp_open(sftp, filename.c_str(),access_type, S_IRWXU);
  if (file == NULL) {
    fprintf(stderr, "Can't open file %s for writing. Error: %s\n",filename.c_str(),ssh_get_error(session));
    return SSH_ERROR;
  }
//  int nwritten = sftp_write(file, buffer, nbyte);
  size_t counter=0;
  size_t nleft=nbyte;
  int nwritten;
  while(counter<nbyte){
      size_t n=min(nleft,(size_t)BUF_SIZE);
      nwritten=sftp_write(file,buffer+counter,n);
//      fprintf(stderr,"n=%d nwritten%d nleft=%d counter=%d\n",n,nwritten,nleft,counter);
      counter+=n;
      nleft-=n;
  }
  if (counter!=nbyte || nleft>0) {
    fprintf(stderr, "Error writing data to file %s. Error: %s\n",filename.c_str(),ssh_get_error(session));
    sftp_close(file);
    return SSH_ERROR;
  }
  int rc = sftp_close(file);
  if (rc != SSH_OK) {
    fprintf(stderr, "Can't close the written file %s. Error: %s\n",filename.c_str(),ssh_get_error(session));
    return rc;
  }
  return SSH_OK;
}

