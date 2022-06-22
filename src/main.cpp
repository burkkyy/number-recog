#include "network.hpp"
#include "read_MNIST.hpp"
#include "matrix.hpp"

int main(int argc, char** argv){
	network net;
	ncreate(net, INPUT_LAYER);
	nrand(net);

	char* train_labels;
	char** train_images;
	int num_train_images = read_MNIST(TRAIN_IMAGES, TRAIN_LABELS, train_images, train_labels);
	
	char* test_labels;
	char** test_images;
	int num_test_images = read_MNIST(TEST_IMAGES, TEST_LABELS, test_images, test_labels);

	ntrain(net, train_images, train_labels, 40000);

	ntest(net, test_images, test_labels, 1000);
	
	/**** MEMORY CLEAN UP ***/
	delete[] train_labels;
	for(int i = 0; i < num_train_images; i++){
		delete[] train_images[i];
	}
	delete[] train_images;
	
	delete[] test_labels;
	for(int i = 0; i < num_test_images; i++){
		delete[] test_images[i];
	}
	delete[] test_images;
	
	nfree(net);
        return 0;
}

