init:
	pip3 install opencv-python
	pip3 install tensorflow
	pip3 install mpi4py

img_gen:
	python3 img_gen.py

pass: pass.c
	mpic++ pass.c -o pass

main: main.c
	mpic++ main.c -o main

run_pass:
	mpirun -np 7 ./pass

run:
	mpirun -np 8 ./main 100 200 100

test: test.c
	mpic++ test.c -o test
	mpirun -np 8 ./test 10 10

t:
	mpirun -np 8 ./test 10 10

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o pass main img/* log/*
