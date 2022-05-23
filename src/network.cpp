#include "network.hpp"
#include "read_MNIST.hpp"

float sigmoid(float x){
	return 0.5f * (x / (1 + abs(x)) + 1);
}

float dsigmoid(float x){
	return 1 / (2 * (1 + abs(x) * (1 + abs(x))));
}

void transfer(float* W, float* B, float* X, float* A, int num_W, int num_A){
	for(int i = 0; i < num_A; i++){
		A[i] = 0;
		for(int j = 0; j < num_W; j++){
			A[i] += W[i * num_A + j] * X[j]; 
		}
		A[i] = sigmoid(A[i] + B[i]);
		
	}
}

void forward_prop(network& net){
	transfer(net.W1, net.B1, net.A0, net.A1, INPUT_LAYER, Z1_LAYER);
	transfer(net.W2, net.B2, net.A1, net.A2, Z1_LAYER, Z2_LAYER);
	transfer(net.W3, net.B3, net.A2, net.A3, Z2_LAYER, OUTPUT_LAYER);
}

void init_net(network& net){
	srand((unsigned) time(NULL));

	// Randomly init weights
	for(int i = 0; i < INPUT_LAYER * Z1_LAYER; i++){	
		net.W1[i] = (float)(1 + (rand() % 100));
		//net.W1[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < Z1_LAYER * Z2_LAYER; i++){	
		net.W2[i] = (float)(1 + (rand() % 100));
		//net.W2[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < Z2_LAYER * OUTPUT_LAYER; i++){	
		net.W3[i] = (float)(1 + (rand() % 100));
		//net.W3[i] = (float)rand() / RAND_MAX * 100;
	}

	// Randomly init biases
	for(int i = 0; i < Z1_LAYER; i++){	
		net.B1[i] = (float)(1 + (rand() % 100));
		//net.B1[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < Z2_LAYER; i++){
		net.B2[i] = (float)(1 + (rand() % 100));
		//net.B2[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < OUTPUT_LAYER; i++){
		net.B3[i] = (float)(1 + (rand() % 100));	
	}
}

void print_output(network& net){
	for(int i = 0; i < OUTPUT_LAYER; i++){
		std::cout << "OUTPUT [" << i << "]: " << net.A3[i] << std::endl;
	}
}

void back_prop(network& net, float*& Y){
	// calc dw and db
	float dw[OUTPUT_LAYER][Z2_LAYER];
	float db[OUTPUT_LAYER];
	for(int i = 0; i < OUTPUT_LAYER; i++){
		for(int j = 0; j < Z2_LAYER; j++){
			dw[i][j] = net.A2[j] * dsigmoid(net.W3[i * OUTPUT_LAYER + j] * net.A2[j] + net.B3[i]) * 2 * (net.A3[i] - Y[i]);
			
			P(dw[i][j]);
		}
	}
	P("dw " << dw[0][0]);
	P("w " << net.W3[0]);
}

