#ifndef NET_H
#define NET_H

//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include "matrix.hpp"

#define TRAIN_IMAGES "resources/train/images"
#define TRAIN_LABELS "resources/train/labels"
#define TEST_IMAGES "resources/test/images"
#define TEST_LABELS "resources/test/labels"

#define INPUT_LAYER 784
#define Z1_LAYER 12
//#define Z2_LAYER 12
#define OUTPUT_LAYER 10

#define LEARNING_RATE 0.01

#define P(x) std::cout << x << std::endl

struct network {
	mat* A0;
	
	mat* W1;
	mat* B1;
	mat* Z1;
	mat* A1;

	mat* W2;
	mat* B2;
	mat* Z2;
	mat* A2;
	
	/*
	float* Z3;
	float* A3;
	float* W3;
	float* B3;
	*/
};

void ncreate(network&);
void nfree(network&);
void nrand(network&);
void ntrain(network&, char**, char*, int);
void forward_prop(network&);
void print_output(network&);
void back_prop(network&, double);

#endif

