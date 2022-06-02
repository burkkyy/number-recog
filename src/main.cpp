#include "network.hpp"
#include "read_MNIST.hpp"
#include "matrix.hpp"

int main(int argc, char** argv){
	network net;
	ncreate(net);
	nrand(net);
	
	
	double Y = 2;
	for(int i = 0; i < 20000; i++){
		forward_prop(net);
		back_prop(net, Y);
		if(i % 1000 == 0){
			P("Test " << i << " " << net.A2->elements[2][0]);
		}
	}
	/*
	char* labels;
	char** images;
	int num_images = read_MNIST(TRAIN_IMAGES, TRAIN_LABELS, images, labels);
	*/
	/*** MEMORY CLEAN UP ***/
	/*
	delete[] labels;
	for(int i = 0; i < num_images; i++){
		delete[] images[i];
	}
	delete[] images;
	*/
	nfree(net);
        return 0;
}

