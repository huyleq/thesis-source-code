#include <cstdlib>
#include <cstdio>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>

int main(){
    int server_socket=socket(AF_INET,SOCK_STREAM,0);
    fprintf(stderr,"done creating socket\n");

    struct sockaddr_in server_address;
    server_address.sin_family=AF_INET;
    server_address.sin_port=htons(19002);
    server_address.sin_addr.s_addr=htonl(INADDR_ANY);
    fprintf(stderr,"done preparig address\n");

    bind(server_socket,(struct sockaddr *)&server_address,sizeof(server_address));
    fprintf(stderr,"done binding\n");

    fprintf(stderr,"listening\n");
    listen(server_socket,1);

    socklen_t client_len;
    struct sockaddr_in client_address;
    int client_socket=accept(server_socket,(struct sockaddr *)&client_address,&client_len);
    fprintf(stderr,"accepted\n");

    fprintf(stderr,"sending to client\n");
//    char server_message[256]="You have reach the server";
//    send(client_socket,server_message,sizeof(server_message),0);
    
    int n=5;
    float *x=new float[n];
    for(int i=0;i<n;i++) x[i]=i;

    int niter=5;
    for(int iter=0;iter<niter;iter++){
        fprintf(stderr,"iter %d\n",iter);
        fprintf(stderr,"sending x[0]=%f\n",x[0]);
        send(client_socket,x,n*sizeof(float),0);
        recv(client_socket,x,n*sizeof(float),0);
        fprintf(stderr,"receiving x[0]=%f\n",x[0]);
        if(iter<niter-1) for(int i=0;i<n;i++) x[i]++;
    }
    
    recv(client_socket,x,n*sizeof(float),0);
    for(int i=0;i<n;i++) fprintf(stderr,"%f\n",x[i]);

    delete []x;

    close(server_socket);
    return 0;
}
