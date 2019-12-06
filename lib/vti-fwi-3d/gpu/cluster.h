#ifndef CLUSTER_H
#define CLUSTER_H

#define MAX_FAIL 5

#include <string>
#include <vector>

class Job{
    public:
    Job(int idx,const std::string &id,const std::string &scriptfile,const std::string &outfile,const std::string &gradfile,const std::string state):_jobIdx(idx),_jobId(id),_scriptFile(scriptfile),_outFile(outfile),_gradFile(gradfile),_jobState(state){};
    void setJobId(const std::string &id);
    void setJobState(const std::string &state);
    void printJob();
    int _jobIdx;
    std::string _jobId,_scriptFile,_outFile,_gradFile,_jobState;
};

bool genPBSScript(const std::string &scriptname,const std::string &jobname,const std::string &output,const std::string &command);

std::string submitPBSScript(const std::string &scriptname);

std::string getPBSJobState(Job &job);

bool genScript(const std::string &scriptname,const std::string &jobname,const std::string &output,const std::string &command);

std::string submitScript(const std::string &scriptname);

std::string getJobState(const std::string &jobid);

std::string runCommand(const std::string &command);

float server_job_partition(int njob,std::vector<float> &minute_per_job,std::vector<int> &partition);

#endif
