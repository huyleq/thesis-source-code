#ifndef MYIO_H
#define MYIO_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <omp.h>

using namespace std;

extern int gargc;
extern char **gargv;

template <typename T> T bswap(T val){
 T retVal;
 char *pVal = (char*) &val;
 char *pRetVal = (char*)&retVal;
 int size = sizeof(T);
 for(int i=0;i<size;i++){
     pRetVal[size-1-i] = pVal[i];
 }
 return retVal;
}
    
template <typename T> void bswap(T *y,T *x,int n){
 int i;
 #pragma omp parallel for default(shared) private(i)
 for(i=0;i<n;++i) y[i]=bswap(x[i]);
 return; 
}
    
void myio_init(int,char **);

void myio_close();

string get_s(const string &);

string get_s(const string &,const string &);

bool get_sarray(const string &sa,vector<string> &a,string delimiter=",");

template <class T> T &close_file(T &ios){
 ios.close();
 ios.clear();
 return ios;
}

template <class T> T &open_file(T &ios,const string &file,ios_base::openmode mode=ios_base::in){
 ios.close();
 ios.clear();
 ios.open(file.c_str(),mode);
 return ios;
}

template<typename T> bool get_first(const string &tag,const string &s,T &x){
 string filename=get_s(tag);
 if(filename==" "){
     fprintf(stderr,"please supply tag %s\n",tag.c_str());
     return false;
 }
 else{
  bool found=false;
  string s0=s+"=",ss=" ";
  ifstream in;
  if(!open_file(in,filename)){
   cout<<"cannot open file "<<filename<<endl;
  }
  else{
   string line;
   while(getline(in,line) && !found){
    size_t pos1=line.find(s0),pos2;
    if(pos1!=string::npos){
 	if(line[pos1+s0.size()]=='\"'){
 	 pos2=line.find("\"",pos1+s0.size()+1);
 	 ss=line.substr(pos1+s0.size()+1,pos2-pos1-s0.size()-1);
 	}
 	else{
      pos2=line.find(" ",pos1);
 	 ss=line.substr(pos1+s0.size(),pos2-pos1-s0.size());
 	}
     found=true;
     break;
    }
   }
  }
  x=stod(ss);
  close_file(in);
  return found;
 }
}

template<typename T> bool get_all(const string &tag,const string &s,vector<T> &x){
 string filename=get_s(tag);
 if(filename==" "){
     fprintf(stderr,"please supply tag %s\n",tag.c_str());
     return false;
 }
 else{
  bool found=false;
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
 	if(line[pos1+s0.size()]=='\"'){
 	 pos2=line.find("\"",pos1+s0.size()+1);
 	 ss=line.substr(pos1+s0.size()+1,pos2-pos1-s0.size()-1);
 	}
 	else{
      pos2=line.find(" ",pos1);
 	 ss=line.substr(pos1+s0.size(),pos2-pos1-s0.size());
 	}
    x.push_back(stod(ss));
 	pos1=line.find(s0,pos2);
    found=true;
    }
   }
  }
  close_file(in);
  return found;
 }
}

template <typename T> bool get_array(const string &sa,vector<T> &a){
 string s=get_s(sa);
 if(s==" "){
  string filename=get_s("par");
  s=get_s(filename,sa);
 }
 if(s!=" "){
  int current=s.find(","),previous=0;
  while(current!=string::npos){
      double x=stod(s.substr(previous,current-previous));
      a.push_back(x);
      previous=current+1;
      current=s.find(",",previous);
  }
  double x=stod(s.substr(previous,current-previous));
  a.push_back(x);
  return true;
 }
 else{
  cout<<"cannot find parameter "<<sa<<endl;
  return false;
 }
}

template <typename T> bool get_param(const string &sa,T &a){
 string s=get_s(sa);
 if(s==" "){
  string filename=get_s("par");
  s=get_s(filename,sa);
 }
 if(s!=" "){
  a=stod(s);
  return true;
 }
 else{
  cout<<"cannot find parameter "<<sa<<endl;
  return false;
 }
}

template <typename T1,typename T2>
bool get_param(const string &sa1,T1 &a1,const string &sa2,T2 &a2){
 bool stat1=get_param(sa1,a1);
 bool stat2=get_param(sa2,a2);
 return (stat1 && stat2);
}

template <typename T1,typename T2,typename T3>
bool get_param(const string &sa1,T1 &a1,const string &sa2,T2 &a2,const string &sa3,T3 &a3){
 bool stat1=get_param(sa1,a1);
 bool stat2=get_param(sa2,a2);
 bool stat3=get_param(sa3,a3);
 return ((stat2 && stat2) && stat3);
}

template <typename T1> 
bool to_header(const string &sh,const string &sa1,T1 a1){
 string filename=get_s(sh);
 if(filename==" "){
  cout<<"cannot find header "<<sh<<endl;
  return false;
 }
 else{
  ofstream outfile;
  if(!open_file(outfile,filename,ofstream::app)){
   cout<<"cannot open header "<<sh<<endl;
   close_file(outfile);
   return false;
  }
  else{
   outfile<<"\n"<<sa1<<"="<<a1<<"\n";
   close_file(outfile);
   return true;
  }
 } 
}

template <typename T1,typename T2> 
bool to_header(const string &sh,const string &sa1,T1 a1,
                               const string &sa2,T2 a2){
 bool stat1=to_header(sh,sa1,a1);
 bool stat2=to_header(sh,sa2,a2);
 return (stat1 && stat2);
}

template <typename T1,typename T2,typename T3> 
bool to_header(const string &sh,const string &sa1,T1 a1,
                               const string &sa2,T2 a2,
                               const string &sa3,T3 a3){
 bool stat1=to_header(sh,sa1,a1);
 bool stat2=to_header(sh,sa2,a2);
 bool stat3=to_header(sh,sa3,a3);
 return ((stat1 && stat2) && stat3);
}

template <typename T1,typename T2,typename T3,typename T4> 
bool to_header(const string &sh,const string &sa1,T1 a1,
                               const string &sa2,T2 a2,
                               const string &sa3,T3 a3,
                               const string &sa4,T4 a4){
 bool stat1=to_header(sh,sa1,a1);
 bool stat2=to_header(sh,sa2,a2);
 bool stat3=to_header(sh,sa3,a3);
 bool stat4=to_header(sh,sa4,a4);
 return ((stat1 && stat2) && (stat3 && stat4));
}

template <typename T> bool from_header(const string &sh,const string &sa,T &a){
 string filename=get_s(sh);
 if(filename==" "){
  cout<<"cannot find header "<<sh<<endl;
  return false;
 }
 else{   
  string s=get_s(filename,sa);
  if(s!=" "){
   a=stod(s);
   return true;
  } 
  else{
   cout<<"cannot find parameter "<<sa<<endl;
   return false;
  }
 }
}

template <typename T1,typename T2> 
bool from_header(const string &sh,const string &sa1,T1 &a1,const string &sa2,T2 &a2){
 bool stat1=from_header(sh,sa1,a1);
 bool stat2=from_header(sh,sa2,a2);
 return (stat1 && stat2);
}

template <typename T1,typename T2,typename T3> 
bool from_header(const string &sh,const string &sa1,T1 &a1,const string &sa2,T2 &a2,const string &sa3,T3 &a3){
 bool stat1=from_header(sh,sa1,a1);
 bool stat2=from_header(sh,sa2,a2);
 bool stat3=from_header(sh,sa3,a3);
 return ((stat1 && stat2) && stat3);
}

template <typename T> 
bool write(const string &s,const T *buff,size_t n,string format="native_float",ios_base::openmode mode=ios_base::out){
 string hfile=get_s(s),hfilename=hfile;
 size_t pos=hfile.rfind("/");
 if(pos!=string::npos) hfilename=hfile.substr(pos+1);
 if(hfile==" "){
  cout<<"cannot find header "<<s<<endl;
  return false;
 }
 string path=get_s("datapath");
 if(path==" "){
 string homedir(getenv("HOME"));
 path=get_s(homedir+"/.datapath","datapath");
 }
 if(path==" "){
  cout<<"cannot find datapath"<<endl;
  return false;
 }
 string bfile=path+hfilename+"@";
 ofstream out;
 bool stat;
 if(!open_file(out,bfile,ofstream::binary|mode)){
  cout<<"cannot write to binary file "<<bfile<<" for header "<<hfile<<endl;
  stat=false;
 }
 else{
  if(format=="native_float") out.write((char*)buff,sizeof(T)*n);  
  else{
   T x;
   for(size_t i=0;i<n;++i){
    x=bswap(buff[i]);
	out.write((char*)&x,sizeof(T));
   }
  }
  stat=true;
 }
 close_file(out);
 if(!open_file(out,hfile,ofstream::app)){
  cout<<"cannot open header "<<hfile<<endl;
  stat=false;
 }
 else{
  for(int i=0;i<gargc;++i) out<<gargv[i]<<" ";
  out<<endl;
  out<<"\n"<<"in="<<bfile<<"\n"<<"data_format="<<format<<endl;
  stat=true;
 }
 close_file(out);
 return stat;
}

template <typename T> 
bool writeToHeader(const string &hfile,const T *buff,size_t n,ios_base::openmode mode=ios_base::out){
 string hfilename=hfile;
 size_t pos=hfile.rfind("/");
 if(pos!=string::npos) hfilename=hfile.substr(pos+1);
 string homedir(getenv("HOME"));
 string path=get_s(homedir+"/.datapath","datapath");
 if(path==" "){
  cout<<"cannot find datapath"<<endl;
  return false;
 }
 string bfile=path+hfilename+"@";
 ofstream out;
 bool stat;
 if(!open_file(out,bfile,ofstream::binary|mode)){
  cout<<"cannot write to binary file "<<bfile<<" for header "<<hfile<<endl;
  stat=false;
 }
 else{
  out.write((char*)buff,sizeof(T)*n);  
  stat=true;
 }
 close_file(out);
 if(!open_file(out,hfile,ofstream::app)){
  cout<<"cannot open header "<<hfile<<endl;
  stat=false;
 }
 else{
  out<<"\n"<<"in="<<bfile<<"\n"<<"data_format=native_float"<<endl;
  for(int i=0;i<gargc;++i) out<<gargv[i]<<" ";
  out<<endl;
  stat=true;
 }
 close_file(out);
 return stat;
}

template <typename T> 
bool write(const string &s,const T *buff,size_t n,ios_base::openmode mode){
 string hfile=get_s(s);
 if(hfile==" "){
  cout<<"cannot find header "<<s<<endl;
  return false;
 }
 return writeToHeader(hfile,buff,n,mode);
}

template <typename T> bool readFromHeader(const string &hfile,T *buff,size_t n,size_t pos=0){
 string bfile=get_s(hfile,"in");
 if(bfile==" ") return false;
 ifstream in;
 bool stat;
 if(!open_file(in,bfile,ofstream::binary)){
  cout<<"cannot read binary file "<<bfile<<" from header "<<hfile<<endl;
  stat=false;
 }
 else{
  in.seekg(sizeof(T)*pos);
  string format=get_s(hfile,"data_format");
  if(format=="native_float" || format=="native_double") in.read((char*)buff,sizeof(T)*n);
  else{
   T x;
   for(size_t i=0;i<n;++i){
	in.read((char*)&x,sizeof(T));
	buff[i]=bswap(x);
   }
  }
  close_file(in);
  stat=true;
 }
 return stat;
}

template <typename T> bool read(const string &s,T *buff,size_t n,size_t pos=0){
 string hfile=get_s(s);
 if(hfile==" "){
  cout<<"cannot find header "<<s<<endl;
  return false;
 }
 return readFromHeader(hfile,buff,n,pos);
}

#endif
