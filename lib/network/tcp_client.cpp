#include <cstdlib>
#include <cstdio>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>
#include <errno.h>

int main(){
    int network_socket=socket(AF_INET,SOCK_STREAM,0);
    fprintf(stderr,"done creating socket\n");

    struct sockaddr_in server_address;
    server_address.sin_family=AF_INET;
    server_address.sin_port=htons(22);
    server_address.sin_addr.s_addr=inet_addr("171.67.166.22"); //oas
//    server_address.sin_addr.s_addr=inet_addr("171.66.97.72"); //ln02
    fprintf(stderr,"done preparing address\n");

    fprintf(stderr,"trying to connect\n");
    int connection_status=connect(network_socket,(struct sockaddr *)&server_address,sizeof(server_address));
    int err=errno;
    if(err==EADDRNOTAVAIL) fprintf(stderr,"The specified address is not available from the local machine.\n");
    else if(err==EAFNOSUPPORT) fprintf(stderr,"The address family is not supported.\n");
    else if(err==EALREADY) fprintf(stderr,"The socket descriptor socket is marked nonblocking, and a previous connection attempt has not completed.\n");
    else if(err==EBADF) fprintf(stderr,"The socket parameter is not a valid socket descriptor.\n");
    else if(err==ECONNREFUSED) fprintf(stderr,"The connection request was rejected by the destination host.\n");
    else if(err==EFAULT) fprintf(stderr,"Using address and address_len would result in an attempt to copy the address into a portion of the caller's address space to which data cannot be written.\n");
    else if(err==EINTR) fprintf(stderr,"The attempt to establish a connection was interrupted by delivery of a signal that was caught. The connection will be established asynchronously.\n");
    else if(err==EINVAL) fprintf(stderr,"The address_len parameter is not a valid length.\n");
    else if(err==EIO) fprintf(stderr,"There has been a network or a transport failure.\n");
    else if(err==EISCONN) fprintf(stderr,"The socket descriptor socket is already connected.\n");
    else if(err==ENETUNREACH) fprintf(stderr,"The network cannot be reached from this host.\n");
    else if(err==ENOTSOCK) fprintf(stderr,"The descriptor refers to a file, not a socket.\n");
    else if(err==EOPNOTSUPP) fprintf(stderr,"The socket parameter is not of type SOCK_STREAM.\n");
    else if(err==EPERM) fprintf(stderr,"connect() caller was attempting to extract a user's identity and the caller's process was not verified to be a server. To be server-verified, the caller's process must have permission to the BPX.SERVER profile (or superuser and BPX.SERVER is undefined) and have called either the __passwd() or pthread_security_np() services before calling connect() to propagate identity.\n");
    else if(err==EPROTOTYPE) fprintf(stderr,"The protocol is the wrong type for this socket.\n");
    else if(err==ETIMEDOUT) fprintf(stderr,"The connection establishment timed out before a connection was made.\n");
    else if(err==ENODATA) fprintf(stderr,"No data available.\n");

    if(connection_status==-1) fprintf(stderr,"error making connection to remote server, error no %d\n",err);

    fprintf(stderr,"receiving\n");
//    char server_response[256];
//    recv(network_socket,&server_response,sizeof(server_response),0);
//    fprintf(stderr,"the server says %s\n",server_response);

    int n=5;
    float *x=new float[n];
   
    int niter=5;
    for(int iter=0;iter<niter;iter++){
        fprintf(stderr,"iter %d\n",iter);
        recv(network_socket,x,n*sizeof(float),0);
        fprintf(stderr,"receiving x[0]=%f\n",x[0]);
        for(int i=0;i<n;i++) x[i]++;
        fprintf(stderr,"sending x[0]=%f\n",x[0]);
        send(network_socket,x,n*sizeof(float),0);
    }
    
    for(int i=0;i<n;i++) fprintf(stderr,"%f\n",x[i]);
    
    delete []x;

    close(network_socket);

    return 0;
}
