CC=mpicc #On CS machines
#CC=cc #On Cray

EXES = Jacobi2D-BlockParallel-MPI-VERT

all:  Jacobi2D-BlockParallel-MPI-VERT

#Jacobi2D-BlockParallel-MPI:  Jacobi2D-BlockParallel-MPI.test.c
#	$(CC) -o Jacobi2D-BlockParallel-MPI Jacobi2D-BlockParallel-MPI.test.c -O3

Jacobi2D-BlockParallel-MPI-VERT: Jacobi2D-BlockParallel-MPI-VERT.test.c
	$(CC) -o Jacobi2D-BlockParallel-MPI-VERT Jacobi2D-BlockParallel-MPI-VERT.test.c -O3


clean:
	rm -f $(EXES)


