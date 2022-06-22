CC=g++
HEADERS=include/
LIBS=`pkg-config --cflags --libs opencv4`

bin/train: src/train.cpp lib/*
	mkdir -p $(dir $@)
	$(CC) $^ -I $(HEADERS) -o $@ $(LIBS)

bin/gui: src/gui.cpp lib/*
	mkdir -p $(dir $@)
	$(CC) $^ -I $(HEADERS) -o $@ $(LIBS)

bin/test: src/test.cpp lib/*
	mkdir -p $(dir $@)
	$(CC) $^ -I $(HEADERS) -o $@ $(LIBS)

.PHONY: all
all: bin/train bin/gui

.PHONY: test
test: bin/test

