#include "network.hpp"

void forward_prop(network& net){
	apply_func(sigmoid, net.A0, net.A0);
	
	mat* temp1 = dot(net.W1, net.A0);
	add(net.B1, temp1, net.Z1);
	apply_func(sigmoid, net.Z1, net.A1);

	mat* temp2 = dot(net.W2, net.A1);
	add(net.B2, temp2, net.Z2);
	apply_func(sigmoid, net.Z2, net.A2);
	//softmax(net);

	mfree(temp1);
	mfree(temp2);
}

void back_prop(network& net){
	// output layer
	mat* delta_A2 = sub(net.A2, net.Y);
	mat* A1_trans = transpose(net.A1);
	
	dot(delta_A2, A1_trans, net.dW2);
	scale(LEARNING_RATE, net.dW2, net.dW2);
	sub(net.W2, net.dW2, net.W2);
	
	scale(LEARNING_RATE, delta_A2, net.dB2);
	sub(net.B2, net.dB2, net.B2);
	
	// input layer
	mat* dactivation = mcopy(net.A1);
	mfill(dactivation, 1);
	sub(dactivation, net.A1, dactivation);
	multiply(dactivation, net.A1, dactivation);
	
	mat* W2_trans = transpose(net.W2);
	mat* delta_A1 = dot(W2_trans, delta_A2);
	multiply(delta_A1, dactivation, delta_A1);

	mat* A0_trans = transpose(net.A0);
	
	dot(delta_A1, A0_trans, net.dW1);
	scale(LEARNING_RATE, net.dW1, net.dW1);
	sub(net.W1, net.dW1, net.W1);

	add(net.dB1, delta_A1, net.dB1);
	scale(LEARNING_RATE, net.dB1, net.dB1);
	sub(net.B1, net.dB1, net.B1);
	
	// memory clean up
	mfree(delta_A2);
	mfree(A1_trans);
	
	mfree(dactivation);
	mfree(W2_trans);
	mfree(delta_A1);
	mfree(A0_trans);
}

void update_net(network& net){
	scale(LEARNING_RATE, net.dW2, net.dW2);
	sub(net.W2, net.dW2, net.W2);
	mfill(net.dW2, 0);

	scale(LEARNING_RATE, net.dB2, net.dB2);
	sub(net.B2, net.dB2, net.B2);
	mfill(net.dB2, 0);

	scale(LEARNING_RATE, net.dW1, net.dW1);
	sub(net.W1, net.dW1, net.W1);
	mfill(net.dW1, 0);

	scale(LEARNING_RATE, net.dB1, net.dB1);
	sub(net.B1, net.dB1, net.B1);
	mfill(net.dB1, 0);
}

void ntrain(network& net, char** images, char* labels, int itter){
	int guess = 0;
	double correct = 0, acc = 0;

	for(int epoch = 0; epoch < 100; epoch++){
		for(int i = 0; i < itter; i++){
			for(int j = 0; j < INPUT_LAYER; j++){
				net.A0->elements[j][0] = images[i][j];
			}

			forward_prop(net);
			hot_encode_y(net, (int)(labels[i]));
			back_prop(net);
			
			guess = argmax(net);
			if(guess == (int)labels[i]){ correct++; }
		}

		if(epoch % 10 == 0){
			acc = correct / itter * 10;
			P("----------");
			P("epoch: " << epoch);
			P("Accuracy: " << acc << "%");
			correct = 0;
		}
	}
}

void ntest(network& net, char** images, char* labels, int itter){
	int guess = 0;
	int correct = 0;
	for(int i = 0; i < itter; i++){
		for(int j = 0; j < INPUT_LAYER; j++){
			net.A0->elements[j][0] = images[i][j];
		}
		
		forward_prop(net);
		
		guess = argmax(net);
		if(guess == (int)labels[i]){ correct++; }

		// P("network guess: " << guess << " Actual " << (int)labels[i]);
	}
	P("Out of " << itter << " guesses, the network got " << correct << " correct");
}

