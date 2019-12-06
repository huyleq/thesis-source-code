#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>
#include <numeric>
#include <vector>

#include "myio.h"

using namespace std;

int gargc;
char **gargv;

void myio_init(int argc,char **argv){
 gargc=argc;
 gargv=argv;
 for(int i=0;i<gargc;++i) fprintf(stderr,"%s ",gargv[i]);
 fprintf(stderr,"\n");
 return;
}

void myio_close(){
 gargc=0;
 gargv=0;
 return;
}

string get_s(const string &s){
 string s0=s+"=",ss=" ",s1;
 for(int i=0;i<gargc;++i){
  s1=gargv[i];
  if(s1.substr(0,s0.size())==s0){
   ss=s1.substr(s0.size());
   break;
  }
 }
// if(ss.compare(" ")==0) fprintf(stderr,"cannot find argument %s\n",s.c_str());
 return ss;
}

string get_s(const string &filename,const string &s){
 string s0=s+"=",ss=" ";
 ifstream in;
 if(!open_file(in,filename)){
  cout<<"cannot open file "<<filename<<endl;
 }
 else{
  string line;
  while(getline(in,line)){
   size_t pos1=line.find(s0),pos2;
   while(pos1!=string::npos){
    if(pos1==0 || (pos1>0 && !isalnum(line[pos1-1]))){
 	 if(line[pos1+s0.size()]=='\"'){
 	  pos2=line.find("\"",pos1+s0.size()+1);
 	  ss=line.substr(pos1+s0.size()+1,pos2-pos1-s0.size()-1);
 	 }
 	 else{
      pos2=line.find(" ",pos1);
 	  ss=line.substr(pos1+s0.size(),pos2-pos1-s0.size());
 	 }
    }
    else{
     pos2=pos1+s0.size();
    }
	pos1=line.find(s0,pos2);
   }
  }
 }
 close_file(in);
 if(ss.compare(" ")==0) fprintf(stderr,"cannot find parameter %s\n",s.c_str());
 return ss;
}

bool get_sarray(const string &sa,vector<string> &a,string delimiter){
 string s=get_s(sa);
 if(s==" "){
  string filename=get_s("par");
  s=get_s(filename,sa);
 }
 if(s!=" "){
  int current=s.find(delimiter),previous=0;
  while(current!=string::npos){
      string x=s.substr(previous,current-previous);
      a.push_back(x);
      previous=current+1;
      current=s.find(delimiter,previous);
  }
  string x=s.substr(previous,current-previous);
  a.push_back(x);
  return true;
 }
 else{
  cout<<"cannot find parameter "<<sa<<endl;
  return false;
 }
}

