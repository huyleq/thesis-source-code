#ifndef CEES_JOB_SUBMIT_H
#define CEES_JOB_SUBMIT_H

#include <string>
#include <vector>
#include "cluster.h"

#define NMAX_JOB 150

void cees_job_init(std::vector<int> &shotid,int icall,const std::string &command,const std::string &workdir,std::vector<Job> &jobs);

void cees_job_submit(Job &job);

void cees_job_submit(std::vector<Job> &jobs);

void cees_job_collect(float *fg,float *temp_fg,size_t nelem,vector<Job> &jobs);

int cees_get_num_job(const std::string &state);

void objFuncGradientCij_cees_cluster(float *fgcij,int nx,int ny,int nz,std::vector<int> &shotid,float pct,int max_shot_per_job,int icall,const std::string &command,const std::string &workdir);

#endif
