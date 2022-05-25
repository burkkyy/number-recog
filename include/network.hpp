#ifndef NET_H
#define NET_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define TRAIN_IMAGES "resources/train/images"
#define TRAIN_LABELS "resources/train/labels"
#define TEST_IMAGES "resources/test/images"
#define TEST_LABELS "resources/test/labels"

#define INPUT_LAYER 784
#define Z1_LAYER 12
#define Z2_LAYER 12
#define OUTPUT_LAYER 10

#define P(x) std::cout << x << std::endl

struct network {
	float* A0;
	
	float* Z1;
	float* A1;
	float* W1;
	float* B1;

	float* Z2;
	float* A2;
	float* W2;
	float* B2;
	
	float* Z3;
	float* A3;
	float* W3;
	float* B3;
};

float activation(float);
float dactivation(float);
void transfer(float*, float*, float*, float*, int, int);
void forward_prop(network&);
void init_net(network&);
void print_output(network&);
void back_prop(network&, float*&);

#endif

