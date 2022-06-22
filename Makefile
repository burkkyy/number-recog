CC=g++
HEADERS=include/
LIBS=`pkg-config --cflags --libs opencv4`

bin/main: src/main.cpp lib/*
	mkdir -p $(dir $@)
	$(CC) $^ -I $(HEADERS) -o $@ $(LIBS)

