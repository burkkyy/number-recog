#include "network.hpp"

// ReLU
double ReLU(double x){
	if(x > 0){
		return x;
	}
	return 0.0;
}

// derivative of ReLU
double dReLU(double x){
	if(x >= 0){
		return 1.0;
	}
	return 0.0;
}

// sigmoid
double sigmoid(double x){
	return 1.0 / (1.0 + exp(x * -1.0));
}

// derivative of sigmoid
double dsigmoid(double x){
	return sigmoid(x) * (1.0 - sigmoid(x));
}

void softmax(network& net){
	double sum = 0;
	double m = -INFINITY;
	
	for (int i = 0; i < 10; i++){
		m = std::max(m, net.Z2->elements[i][0]);
	}

	for (int j = 0; j < 10; j++){
		sum += std::exp(net.Z2->elements[j][0] - m);
	}

	for (int i = 0; i < 10; i++){
		net.A2->elements[i][0] = std::exp2f(net.Z2->elements[i][0] - m) / sum;
	}
}

void print_output(network& net){
	for(int i = 0; i < OUTPUT_LAYER; i++){
		std::cout << "Output Layer[" << i << "]: " << net.A2->elements[i][0] << std::endl;
	}
}

double cost(network& net){
	double cost = 0;
	for(int i = 0; i < OUTPUT_LAYER; i++){
		cost += (net.Y->elements[i][0] - net.A2->elements[i][0]) *\
			(net.Y->elements[i][0] - net.A2->elements[i][0]);
	}
	return cost / 10;
}

double dcost(network& net, int i){
	return -2 * (net.Y->elements[i][0] - net.A2->elements[i][0]);
}


void hot_encode_y(network& net, int y){
	for(int i = 0; i < net.Y->rows; i++){ net.Y->elements[i][0] = 0; }
	net.Y->elements[y][0] = 1;
}

void ncreate(network& net, int input_layer){
	net.A0 = mcreate(input_layer, 1);
	
	net.W1 = mcreate(Z1_LAYER, input_layer);
	net.B1 = mcreate(Z1_LAYER, 1);
	net.Z1 = mcreate(Z1_LAYER, 1);
	net.A1 = mcreate(Z1_LAYER, 1);
	
	net.W2 = mcreate(OUTPUT_LAYER, Z1_LAYER);
	net.B2 = mcreate(OUTPUT_LAYER, 1);
	net.Z2 = mcreate(OUTPUT_LAYER, 1);
	net.A2 = mcreate(OUTPUT_LAYER, 1);

	net.dW1 = mcreate(net.W1->rows, net.W1->cols);
	net.dB1 = mcreate(net.B1->rows, net.B1->cols);
	net.dW2 = mcreate(net.W2->rows, net.W2->cols);
	net.dB2 = mcreate(net.B2->rows, net.B2->cols);
	net.Y = mcreate(OUTPUT_LAYER, 1);
}

void nfree(network& net){
	mfree(net.A0);
	
	mfree(net.W1);
	mfree(net.B1);
	mfree(net.Z1);
	mfree(net.A1);
	
	mfree(net.W2);
	mfree(net.B2);
	mfree(net.Z2);
	mfree(net.A2);

	mfree(net.dW1);
	mfree(net.dB1);
	mfree(net.dW2);
	mfree(net.dB2);
	mfree(net.Y);
}

void nrand(network& net){
	mrand(net.W1);
	//mrand(net.B1);
	mrand(net.W2);
	//mrand(net.B2);
}

