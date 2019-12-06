#include "lap3d.h"

export void lap3d(uniform int n,uniform int nx,uniform int nxy,uniform float lapu[],uniform float u[]){
    foreach(i=0...n){
        lapu[i]=u[i+1]+u[i-1]+u[i+nx]+u[i-nx]+u[i+nxy]+u[i-nxy]-6.f*u[i];
    }
}

