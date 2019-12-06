#!/usr/bin/env tcsh

make test1c.x
./test1c.x 0 NLCG >& test1c0CG-MPR.log
./test1c.x 1 NLCG >& test1c1CG-MPR.log
./test1c.x 2 NLCG >& test1c2CG-MPR.log
./test1c.x 3 NLCG >& test1c3CG-MPR.log
./test1c.x 4 NLCG >& test1c4CG-MPR.log
./test1c.x 5 NLCG >& test1c5CG-MPR.log

make test1f.x
./test1c.x 0 STEEPEST >& test1c0ST.log
./test1c.x 1 STEEPEST >& test1c1ST.log
./test1c.x 2 STEEPEST >& test1c2ST.log
./test1c.x 3 STEEPEST >& test1c3ST.log
./test1c.x 4 STEEPEST >& test1c4ST.log
./test1c.x 5 STEEPEST >& test1c5ST.log

./test1c.x 0 LBFGS >& test1c0LB.log
./test1c.x 1 LBFGS >& test1c1LB.log
./test1c.x 2 LBFGS >& test1c2LB.log
./test1c.x 3 LBFGS >& test1c3LB.log
./test1c.x 4 LBFGS >& test1c4LB.log
./test1c.x 5 LBFGS >& test1c5LB.log
#
#./test1f.x 0 >& test1f0.log
#./test1f.x 1 >& test1f1.log
#./test1f.x 2 >& test1f2.log
#./test1f.x 3 >& test1f3.log
#./test1f.x 4 >& test1f4.log
#./test1f.x 5 >& test1f5.log
