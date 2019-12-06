#ifndef MYLIB_H
#define MYLIB_H

#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>

#define NTHREAD 16

template<typename T>
inline void rand(T *x,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) x[i]=(float)rand()/RAND_MAX;
 return;
}

template<typename T>
inline void transp(T *y, T *x,size_t n1,size_t n2){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i1=0;i1<n1;++i1){
  for(size_t i2=0;i2<n2;++i2){
   y[i2+n2*i1]=x[i1+n1*i2];
  }
 }
 return;
}

template<typename T1,typename T2,typename T3>
inline void add(T1 *ab,const T2 *a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]+b[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void subtract(T1 *ab,const T2 *a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]-b[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void multiply(T1 *ab,const T2 *a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]*b[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void divide(T1 *ab,const T2 *a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]/b[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void divide(T1 *ab,const T2 a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a/b[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void reverse_multiply(T1 *ab,const T2 *a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]*b[n-1-i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void scale_add(T1 *ab,const T2 *a,T3 b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]+=a[i]*b;
 return;
}

template<typename T1,typename T2,typename T3>
inline void scale_add(T1 *ab,T2 b,const T3 *a,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]+=a[i]*b;
 return;
}

template<typename T1,typename T2>
inline void scale(T1 *a,T2 b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) a[i]*=b;
 return;
}

template<typename T1,typename T2,typename T3>
inline void scale(T1 *ab,const T2 *a,T3 b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]*b;
 return;
}

template<typename T1,typename T2,typename T3>
inline void scale(T1 *ab,T2 a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a*b[i];
 return;
}

template<typename T1,typename T2,typename T3,typename T4,typename T5>
inline void lin_comb(T1 *z,T2 a,const T3 *x,T4 b,const T5 *y,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) z[i]=a*x[i]+b*y[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void shift(T1 *ab,const T2 *a,T3 b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a[i]+b;
 return;
}

template<typename T1,typename T2,typename T3>
inline void shift(T1 *ab,T2 a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) ab[i]=a+b[i];
 return;
}

template<typename T1,typename T2,typename T3>
inline void pow(T1 *ab,const T2 *a,T3 b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) ab[i]=pow(a[i],b);
 return;
}

template<typename T1,typename T2,typename T3>
inline void pow(T1 *ab,T2 a,const T3 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) ab[i]=pow(a,b[i]);
 return;
}

template<typename T1,typename T2>
inline void sqrt(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) a[i]=sqrt(b[i]);
 return;
}

template<typename T1,typename T2>
inline void cbrt(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) a[i]=cbrt(b[i]);
 return;
}

template<typename T1,typename T2>
inline void exp(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) a[i]=exp(b[i]);
 return;
}

template<typename T1,typename T2>
inline void reciprocal(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) a[i]=1./b[i];
 return;
}

template<typename T1,typename T2>
inline void log(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) a[i]=log(b[i]);
 return;
}

template<typename T1,typename T2>
inline void equate(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) a[i]=b[i];
 return;
}

template<typename T1,typename T2>
inline void mynegate(T1 *a,const T2 *b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD) 
 for(size_t i=0;i<n;++i) a[i]=-b[i];
 return;
}

template<typename T1,typename T2>
inline void set(T1 *a,T2 b,size_t n){
 #pragma omp parallel for num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) a[i]=b;
 return;
}

template<typename T>
inline double sum(const T *a,size_t n){
 double s=0.;
 #pragma omp parallel for reduction(+:s) num_threads(NTHREAD)
 for(size_t i=0;i<n;++i) s+=a[i];
 return s;
}

template<typename T1,typename T2>
inline double dot_product(const T1 *a,const T2 *b,size_t n){
 double s=0.;
 #pragma omp parallel for num_threads(NTHREAD) reduction(+:s) 
 for(size_t i=0;i<n;++i) s+=a[i]*b[i];
 return s;
}

template<typename T>
inline T min(const T *a,size_t n){
 T m=a[0];
 for(size_t i=1;i<n;++i){
     if(!std::isfinite(a[i])){
       fprintf(stderr,"not a finite number at index %zd\n",i);
       break;
     }
     else if(a[i]<m) m=a[i];
 }
 return m;
}

template<typename T>
inline T max(const T *a,size_t n){
 T m=a[0];
 for(size_t i=1;i<n;++i){
     if(!std::isfinite(a[i])){
       fprintf(stderr,"not a finite number at index %zd\n",i);
       break;
     }
     else if(a[i]>m) m=a[i];
 }
 return m;
}

template<typename T>
inline T max_abs(const T *a,size_t n){
 T m=fabs(a[0]);
 for(size_t i=1;i<n;++i){
     if(!std::isfinite(a[i])){
       fprintf(stderr,"not a finite number at index %zd\n",i);
       break;
     }
     else if(fabs(a[i])>m) m=fabs(a[i]);
 }
 return m;
}

template <typename T> 
inline size_t sign(T val){
 return (T(0) < val) - (val < T(0));
}

template <typename T> void print(const std::string &s,const T *x,size_t n){
    std::cout<<s<<std::endl;
    for(size_t i=0;i<n;i++) std::cout<<"i="<<i<<" value="<<x[i]<<std::endl;
    std::cout<<std::endl;
    return;
}

#endif
