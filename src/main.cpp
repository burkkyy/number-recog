#include "network.hpp"
#include "read_MNIST.hpp"

int main(int argc, char** argv){
	network net;
	
	// input layer
	net.A0 = new float[INPUT_LAYER]();

        // first hidden layer
	net.Z1 = new float[Z1_LAYER]();
        net.A1 = new float[Z1_LAYER]();
        net.W1 = new float[INPUT_LAYER * Z1_LAYER]();
	net.B1 = new float[Z1_LAYER]();

	// second hidden layer
	net.Z2 = new float[Z2_LAYER]();
	net.A2 = new float[Z2_LAYER]();
	net.W2 = new float[Z1_LAYER * Z2_LAYER]();
	net.B2 = new float[Z2_LAYER]();
        
	// output layer
	net.Z3 = new float[OUTPUT_LAYER]();
        net.A3 = new float[OUTPUT_LAYER]();
        net.W3 = new float[Z2_LAYER * OUTPUT_LAYER]();
        net.B3 = new float[OUTPUT_LAYER]();
        float* Y = new float[10]();
        
	// set random values for the weights and biases
	init_net(net);
	forward_prop(net);
	print_output(net);
	
	char* labels;
	char** images;
	int num_images = read_MNIST(TRAIN_IMAGES, TRAIN_LABELS, images, labels);
	
	for(int i = 0; i < 784; i++){
		net.A0[i] = images[0][i];
	}
	std::cout << "label for first image: " << (int)labels[0] << std::endl;
	
	forward_prop(net);
	print_output(net);
	for(int i = 0; i < 100; i++){
		Y[(int)labels[i]] = 1;
		back_prop(net, Y);
	}
	/*** MEMORY CLEAN UP ***/
	delete[] labels;
	for(int i = 0; i < num_images; i++){
		delete[] images[i];
	}
	delete[] images;

        delete[] net.A0;

	delete[] net.Z1;
        delete[] net.A1;
        delete[] net.W1;
        delete[] net.B1;
        
	delete[] net.Z2;
	delete[] net.A2;
        delete[] net.W2;
        delete[] net.B2;

	delete[] net.Z3;
        delete[] net.A3;
	delete[] net.W3;
        delete[] net.B3;
        delete[] Y;
        return 0;
}

