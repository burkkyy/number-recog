#include "network.hpp"
#include "read_MNIST.hpp"

// sigmoid
float activation(float x){
	return 0.5f * (x / (1 + abs(x)) + 1);
}

// d sigmoid
float dactivation(float x){
	return 1 / (2 * (1 + abs(x) * (1 + abs(x))));
}

/*
// RElu
float activation(float x){
	return std::max(0.0f, x);
}

// d RElu
float dactivation(float x){
	if(x >= 0){
		return 1;
	}
	return 0;
}

void transfer(float* W, float* B, float* X, float* A, int num_W, int num_A){
	for(int i = 0; i < num_A; i++){
		A[i] = 0;
		for(int j = 0; j < num_W; j++){
			A[i] += W[i * num_A + j] * X[j]; 
		}
		A[i] = activation(A[i] + B[i]);
	}
}
*/

void forward_prop(network& net){
	for(int i = 0; i < Z1_LAYER; i++){
		net.Z1[i] = net.B1[i];
		for(int j = 0; j < INPUT_LAYER; j++){
			net.Z1[i] += net.W1[i * INPUT_LAYER + j] * net.A0[j];
		}
		net.A1[i] = activation(net.Z1[i]);
	}
	
	/*
	for(int i = 0; i < Z2_LAYER; i++){
		net.Z2[i] = net.B2[i];
		for(int j = 0; j < Z1_LAYER; j++){
			net.Z2[i] += net.W2[i * Z1_LAYER + j] * net.A1[j];
		}
		net.A2[i] = activation(net.Z2[i]);
	}
	
	for(int i = 0; i < OUTPUT_LAYER; i++){
		net.Z3[i] = net.B3[i];
		for(int j = 0; j < Z2_LAYER; j++){
			net.Z3[i] += net.W3[i * Z2_LAYER + j] * net.A2[j];
		}
		net.A3[i] = activation(net.Z3[i]);
	}
	*/

	for(int i = 0; i < OUTPUT_LAYER; i++){
		net.Z2[i] = net.B2[i];
		for(int j = 0; j < Z1_LAYER; j++){
			net.Z2[i] += net.W2[i * Z1_LAYER + j] * net.A1[j];
		}
		net.A2[i] = activation(net.Z2[i]);
	}

	/*
	transfer(net.W1, net.B1, net.A0, net.A1, INPUT_LAYER, Z1_LAYER);
	transfer(net.W2, net.B2, net.A1, net.A2, Z1_LAYER, Z2_LAYER);
	transfer(net.W3, net.B3, net.A2, net.A3, Z2_LAYER, OUTPUT_LAYER);
	*/
}

void init_net(network& net){
	srand((unsigned) time(NULL));

	// Randomly init weights
	for(int i = 0; i < INPUT_LAYER * Z1_LAYER; i++){	
		//net.W1[i] = (float)(1 + (rand() % 1000));
		net.W1[i] = (float)(rand() / (float)(RAND_MAX / 10.0f));
	}

	for(int i = 0; i < Z1_LAYER * OUTPUT_LAYER; i++){
		net.W2[i] = (float)(rand() / (float)(RAND_MAX / 10.0f));
	}
	
	/*
	for(int i = 0; i < Z1_LAYER * Z2_LAYER; i++){	
		//net.W2[i] = (float)(1 + (rand() % 1000));
		net.W2[i] = (float)(rand() / (float)(RAND_MAX / 10.0f));
	}

	for(int i = 0; i < Z2_LAYER * OUTPUT_LAYER; i++){	
		//net.W3[i] = (float)(1 + (rand() % 1000));
		net.W3[i] = (float)(rand() / (float)(RAND_MAX / 10.0f));
	}
	*/
	
	// Randomly init biases
	for(int i = 0; i < Z1_LAYER; i++){	
		//net.B1[i] = (float)(1 + (rand() % 100));
		net.B1[i] = (float)(rand() / (float)(RAND_MAX / 100.0f));
	}
	
	/*
	for(int i = 0; i < Z2_LAYER; i++){
		//net.B2[i] = (float)(1 + (rand() % 100));
		net.B2[i] = (float)(rand() / (float)(RAND_MAX / 100.0f));
	}
	
	for(int i = 0; i < OUTPUT_LAYER; i++){
		net.B3[i] = (float)(rand() / (float)(RAND_MAX / 100.0f));	
	}
	*/
	
	for(int i = 0; i < OUTPUT_LAYER; i++){
		net.B2[i] = (float)(rand() / (float)(RAND_MAX / 100.0f));
	}
}

/*
void print_output(network& net){
	for(int i = 0; i < OUTPUT_LAYER; i++){
		std::cout << "OUTPUT [" << i << "]: " << net.A3[i] << std::endl;
	}
}

float cost_function(network& net, float*& Y){
	float cost = 0.0f;
	for(int i = 0; i < OUTPUT_LAYER; i++){
		cost += (net.A3[i] - Y[i]) * (net.A3[i] - Y[i]);
	}
	return cost;
}

float dcost_function(network& net, float*& Y, int i){
	return 2 * (net.A3[i] - Y[i]);
}
*/

void print_output(network& net){
	for(int i = 0; i < OUTPUT_LAYER; i++){
		std::cout << "OUTPUT [" << i << "]: " << net.A2[i] << std::endl;
	}
}

float cost_function(network& net, float*& Y){
	float cost = 0.0f;
	for(int i = 0; i < OUTPUT_LAYER; i++){
		cost += (net.A2[i] - Y[i]) * (net.A2[i] - Y[i]);
	}
	return cost;
}

float dcost_function(network& net, float*& Y, int i){
	return 2 * (net.A2[i] - Y[i]);
}

void back_prop(network& net, float*& Y){
	/*
	// calc graidents for output layer
	float dw3[OUTPUT_LAYER * Z2_LAYER];
	for(int i = 0; i < OUTPUT_LAYER; i++){
		for(int j = 0; j < Z2_LAYER; j++){
			dw3[i * OUTPUT_LAYER + j] = dcost_function(net, Y, i) * dactivation(net.Z3[i]) * net.A2[j];
		}
	}
	
	float db3[OUTPUT_LAYER];
	for(int i = 0; i < OUTPUT_LAYER; i++){
		db3[i] = dcost_function(net, Y, i) * dactivation(net.Z3[i]);
	}

	// calc graidents for 2nd hidden layer
	// ROUGH WORK
	// dw2 = dcost_func / da3 * da3 / dz3 * dz3 / da2 * da2 / dz2 * dz2 / dw2 
	// dw2 = dcost_function(net, Y, i) * dactivation(net.Z3) * net.W2 * dactivation(net.Z2) * net.A1
	//
	float dw2[Z2_LAYER * Z1_LAYER];
	for(int i = 0; i < Z2_LAYER; i++){
		for(int j = 0; j < Z1_LAYER; j++){
			
		}
	}
	*/

	// calc graidents for output layer
	float dw2[OUTPUT_LAYER * Z1_LAYER];
	for(int i = 0; i < OUTPUT_LAYER; i++){
		for(int j = 0; j < Z1_LAYER; j++){
			dw2[i * Z1_LAYER + j] = dcost_function(net, Y, i) * dactivation(net.Z2[i]) * net.A1[j];
		}
	}
	
	float db2[OUTPUT_LAYER];
	for(int i = 0; i < OUTPUT_LAYER; i++){
		db2[i] = dcost_function(net, Y, i) * dactivation(net.Z2[i]);
	}

	// calc graidents for hidden layer
	// ROUGH WORK
	// dw1 = dcost_func / da2 * da2 / dz2 * dz2 / da1 * da1 / dz1 * dz1 / dw1
	// dw1 = dcost_function(net, Y, i) * dactivation(net.Z2) * net.W2 * dactivation(net.Z1) * net.A1
	//
	// db1 = dcost_func / da2 * da2 / dz2 * dz2 / da1 * da1 / dz1 * dz1 / db1
	// db1 = dcost_function(net, Y, i) * dactivation(net.Z2) * net.W2 * dactivation(net.Z1)
	float dw1[INPUT_LAYER * Z1_LAYER];
	for(int i = 0; i < Z1_LAYER; i++){
		for(int j = 0; j < INPUT_LAYER; j++){
			for(int k = 0; k < OUTPUT_LAYER; k++){
				dw1[i * INPUT_LAYER + j] += dcost_function(net, Y, k) * dactivation(net.Z2[k]) * net.W2[k * Z1_LAYER + i] * dactivation(net.Z1[i]) * net.A1[i];
			}
		}
	}

	float db1[Z1_LAYER];
	for(int i = 0; i < Z1_LAYER; i++){
		for(int j = 0; j < OUTPUT_LAYER; j++){
			db1[i] += dcost_function(net, Y, j) * dactivation(net.Z2[j]) * net.W2[j * Z1_LAYER + i] * dactivation(net.Z1[i]);
		}
	}
	
	for(int i = 0; i < OUTPUT_LAYER; i++){
		net.B2[i] -= db2[i] * LEARNING_RATE;
		for(int j = 0; j < Z1_LAYER; j++){
			net.W2[i * Z1_LAYER + j] -= dw2[i * INPUT_LAYER + j] * LEARNING_RATE;
		}
	}
	
	for(int i = 0; i < Z1_LAYER; i++){
		net.B1[i] -= db1[i] * LEARNING_RATE;
		for(int j = 0; j < INPUT_LAYER; j++){
			net.W1[i * INPUT_LAYER + j] -= dw1[i * INPUT_LAYER + j] * LEARNING_RATE;
		}
	}	
}

