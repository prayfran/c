
CC = gcc
OBJS = timer.o
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3 

EXEC = knap1 knap2 knap3 knap4

all: $(EXEC) 

knap1: knap1.c timer.o
	$(CC) $(SEQFLAGS) -o knap1 knap1.c $(OBJS)

knap2: knap2.c timer.o
	$(CC) $(FLAGS)  -o knap2 knap2.c $(OBJS)

knap3: knap3.c timer.o
	$(CC) $(FLAGS) -o knap3 knap3.c $(OBJS)

knap4: knap4.c timer.o
	$(CC) $(FLAGS) -o knap4 knap4.c $(OBJS)

timer.o: timer.c
	$(CC) -o $@ -c timer.c

clean:
	rm -f $(EXEC) $(OBJS)
