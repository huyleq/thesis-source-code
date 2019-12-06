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
#include <array> 
#include <memory>

#include "sshtunneling.h"

using namespace std;

int authenticate_pubkey(ssh_session session) {
  int rc = ssh_userauth_publickey_auto(session, NULL, NULL);
  if (rc == SSH_AUTH_ERROR) {
     fprintf(stderr, "Authentication failed: %s\n",ssh_get_error(session));
     return SSH_AUTH_ERROR;
  }
  return rc;
}

int main(int argc,char **argv){
    // argv[2] is script file, argv[1] is remote server name
    if(argc>=3){
        // Open session and set options
        cout<<"open ssh session"<<endl;
        ssh_session my_ssh_session = ssh_new();
        if (my_ssh_session == NULL){
            cout<<"cannot open ssh session"<<endl;
            exit(-1);
        }
        cout<<"ssh session opened"<<endl;
        ssh_options_set(my_ssh_session, SSH_OPTIONS_HOST,argv[1]);
        
        // Connect to server
        cout<<"connecting to "<<argv[1]<<endl;
        int rc = ssh_connect(my_ssh_session);
        if (rc != SSH_OK){
          fprintf(stderr, "Error connecting to %s: %s\n",argv[1],ssh_get_error(my_ssh_session));
          ssh_free(my_ssh_session);
          exit(-1);
        }
        cout<<"connect successfully"<<endl;
      
        // Authenticate ourselves
        cout<<"authenticating"<<endl;
        rc = authenticate_pubkey(my_ssh_session);
        if (rc != SSH_AUTH_SUCCESS){
          fprintf(stderr, "Error authenticating with public key: %s\n",ssh_get_error(my_ssh_session));
          ssh_disconnect(my_ssh_session);
          ssh_free(my_ssh_session);
          exit(-1);
        }
        cout<<"authenticate successfully"<<endl;
      
        string scriptfile=argv[2];
        string command=scriptfile;
        string output;
        cout<<"running command "<<command<<endl;
        ssh_run_command(command,my_ssh_session,output);
        cout<<"output from running command "<<command<<" is:"<<endl;
        cout<<output<<endl;
        ssh_disconnect(my_ssh_session);
        ssh_free(my_ssh_session);
    }
    else if(argc==2){
        string scriptfile=argv[1];
        string command=scriptfile;
        array<char, 128> buffer;
        string output;
        unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(),"r"), pclose);
        if(!pipe) throw runtime_error("popen() failed!");
        while(fgets(buffer.data(),buffer.size(),pipe.get())!= nullptr) output+=buffer.data();
    }
    else fprintf(stderr,"Not enough arguments: argv[1] is server, argv[2] is script file\n");
    return 0;
}
