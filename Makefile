CC=g++
HEADERS=include/
LIBS=`pkg-config --cflags --libs opencv4`

bin/main: src/main.cpp lib/read_MNIST.cpp lib/network.cpp lib/matrix.cpp lib/matrix_utilities.cpp
	mkdir -p $(dir $@)
	$(CC) $^ -I $(HEADERS) -o $@ $(LIBS)

