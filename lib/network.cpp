#include "network.hpp"
#include "matrix.hpp"
#include "read_MNIST.hpp"

// ReLU
double ReLU(double x){
	if(x > 0){
		return x;
	}
	return 0;
}

// derivative of ReLU
double dReLU(double x){
	if(x > 0){
		return 1;
	}
	return 0;
}

// sigmoid
double sigmoid(double x){
	return 0.5f * (x / (1.0f + abs(x)) + 1.0f);
}

// derivative of sigmoid
double dsigmoid(double x){
	return sigmoid(x) * (1.0f - sigmoid(x));
}

void forward_prop(network& net){
	// set input layer
	//mfree(net.A0);
	//net.A0 = input;

	mat* temp;
	
	// apply weights
	temp = dot(net.W1, net.A0);
	mmove(net.Z1, temp);
	mfree(temp);
	
	// apply biases
	temp = add(net.Z1, net.B1);
	mmove(net.Z1, temp);
	mfree(temp);
	
	// apply activation func
	temp = apply_func(sigmoid, net.Z1);
	mmove(net.A1, temp);
	mfree(temp);
	
	// apply weights
	temp = dot(net.W2, net.A1);
	mmove(net.Z2, temp);
	mfree(temp);

	// apply biases
	temp = add(net.Z2, net.B2);
	mmove(net.Z2, temp);
	mfree(temp);

	// apply activation func
	temp = apply_func(sigmoid, net.Z2);
	mmove(net.A2, temp);
	mfree(temp);
}

void print_output(network& net){
	for(int i = 0; i < OUTPUT_LAYER; i++){
		std::cout << "Output Layer[" << i << "]: " << net.A2->elements[i][0] << std::endl;
	}
}

double cost(network& net, mat*& Y){
	double cost = 0;
	for(int i = 0; i < OUTPUT_LAYER; i++){
		cost += (net.A2->elements[i][0] - Y->elements[i][0]) * (net.A2->elements[i][0] - Y->elements[i][0]);
	}
	return cost / 2;
}

double dcost(network& net, mat*& Y, int i){
	return net.A2->elements[i][0] - Y->elements[i][0];
}

void back_prop(network& net, double Y){
	/*
	P("W1 " << net.W1->rows << " " << net.W1->cols);
	P("W2 " << net.W2->rows << " " << net.W2->cols);
	P("A1 " << net.A1->rows << " " << net.A1->cols);
	P("A2 " << net.A2->rows << " " << net.A2->cols);
	*/

	mat* one_hot_y = mcreate(10, 1);
	one_hot_y->elements[(int)Y][0] = 1;
	
	mat* dW2 = mcreate(net.W2->rows, net.W2->cols);
	for(int i = 0; i < dW2->rows; i++){
		for(int j = 0; j < dW2->cols; j++){
			/* debugging helper code
			P(i << " " << j);
			P(dcost(net, one_hot_y, i));
			P(dsigmoid(net.Z2->elements[i][0]));
			P(net.A1->elements[j][0]);
			P("=======");
			*/
			dW2->elements[i][j] = \
					      dcost(net, one_hot_y, i) *\
					      dsigmoid(net.Z2->elements[i][0]) *\
					      net.A1->elements[j][0];
		}
	}

	mat* dB2 = mcreate(net.B2->rows, net.B1->cols);
	for(int i = 0; i < dB2->rows; i++){
		dB2->elements[i][0] = \
				      dcost(net, one_hot_y, i) *\
				      dsigmoid(net.Z2->elements[i][0]);	
	}

	mat* dW1 = mcreate(net.W1->rows, net.W1->cols);
	for(int i = 0; i < dW1->rows; i++){
		for(int j = 0; j < dW1->cols; j++){
			for(int k = 0; k < OUTPUT_LAYER; k++){
				dW1->elements[i][j] += \
						       dcost(net, one_hot_y, k) *\
						       dsigmoid(net.Z2->elements[k][0]) *\
						       net.W2->elements[k][i] *\
						       dsigmoid(net.Z1->elements[i][0]) *\
						       net.A1->elements[i][0];		
			}
		}
	}

	mat* dB1 = mcreate(net.B1->rows, net.B1->cols);
	for(int i = 0; i < dB1->rows; i++){
		for(int j = 0; j < OUTPUT_LAYER; j++){
			dB1->elements[i][0] = \
					      dcost(net, one_hot_y, j) *\
					      dsigmoid(net.Z2->elements[j][0]) *\
					      net.W2->elements[j][i] *\
					      dsigmoid(net.Z1->elements[i][0]);
		}
	}

	for(int i = 0; i < net.W2->rows; i++){
		for(int j = 0; j < dW2->cols; j++){
			net.W2->elements[i][j] -= dW2->elements[i][j] * LEARNING_RATE;
		}
	}

	for(int i = 0; i < net.B2->rows; i++){
		net.B2->elements[i][0] -= dB2->elements[i][0] * LEARNING_RATE;
	}

	for(int i = 0; i < net.W1->rows; i++){
		for(int j = 0; j < net.W1->cols; j++){
			net.W1->elements[i][j] -= dW1->elements[i][j] * LEARNING_RATE;
		}
	}

	for(int i = 0; i < net.B1->rows; i++){
		net.B1->elements[i][0] -= dB1->elements[i][0] * LEARNING_RATE;
	}

	mfree(one_hot_y);
	mfree(dW2);
	mfree(dB2);
	mfree(dW1);
	mfree(dB1);
}

void ntrain(network& net, char** images, char* labels, int itter){
	for(int i = 0; i < itter; i++){
		for(int j = 0; j < INPUT_LAYER; j++){
			net.A0->elements[j][0] = images[i][j];
		}
		forward_prop(net);
		back_prop(net, (double)(labels[i]));	
	}
}


void ncreate(network& net){
	net.A0 = mcreate(INPUT_LAYER, 1);
	
	net.W1 = mcreate(Z1_LAYER, INPUT_LAYER);
	net.B1 = mcreate(Z1_LAYER, 1);
	net.Z1 = mcreate(Z1_LAYER, 1);
	net.A1 = mcreate(Z1_LAYER, 1);
	
	net.W2 = mcreate(OUTPUT_LAYER, Z1_LAYER);
	net.B2 = mcreate(OUTPUT_LAYER, 1);
	net.Z2 = mcreate(OUTPUT_LAYER, 1);
	net.A2 = mcreate(OUTPUT_LAYER, 1);
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
}

void nrand(network& net){
	mrand(net.W1);
	mrand(net.W2);
}

