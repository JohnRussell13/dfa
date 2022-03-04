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
	mpirun -np 15 ./main 100 50 30 10

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o pass main img/* log/*
