#ifndef NET_H
#define NET_H

//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <math.h>
#include "matrix.hpp"

#define TRAIN_IMAGES "resources/train/images"
#define TRAIN_LABELS "resources/train/labels"
#define TEST_IMAGES "resources/test/images"
#define TEST_LABELS "resources/test/labels"

#define INPUT_LAYER 784
#define Z1_LAYER 40
#define OUTPUT_LAYER 10
#define LEARNING_RATE 0.1

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

	mat* dW1;
	mat* dB1;
	mat* dW2;
	mat* dB2;
	mat* Y;
};

double ReLU(double);
double dReLU(double);

double sigmoid(double);
double dsigmoid(double);

void softmax(network&);

double cost(network&);
double dcost(network&, int);

void hot_encode_y(network&, int);

void ncreate(network&, int);
void nfree(network&);
void nrand(network&);

void forward_prop(network&);
void back_prop(network&);
void update_net(network&);
void print_output(network&);

void ntrain(network&, char**, char*, int);
void ntest(network&, char**, char*, int);

#endif

