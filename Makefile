CC=g++
LIBS=`pkg-config --cflags --libs opencv4`

bin/main: main.cpp
	mkdir -p $(dir $@)
	$(CC) $^ -o $@ $(LIBS)

