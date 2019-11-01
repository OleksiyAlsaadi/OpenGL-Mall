OBJS = main.c

CC = g++

COMPILER_FLAGS = -w

LINKER_FLAGS = -lSOIL -lglut -lGL -lGLEW -std=c++11

all : $(OBJS)
	$(CC) $(OBJS) $(COMPILER_FLAGS) $(LINKER_FLAGS) 
