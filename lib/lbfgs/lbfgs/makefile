FC=gfortran
CC=g++ -std=c++11

.SECONDARY:

%.o: %.f
	$(FC) -c $< -o $@

%.o: %.cpp
	$(CC) -c $< -o $@

test%c.x: test%c.o lbfgs_c.o steepest.o nlcg.o la.o lsearch.o functions.o
	$(CC) $^ -o $@

test%f.x: test%f.o lbfgs_f.o functions.o
	$(CC) $^ -o $@ -lgfortran
	
clean:
	rm -f *.x *.o  *.log

