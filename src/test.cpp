#include "network.hpp"
#include "read_MNIST.hpp"
#include "matrix.hpp"

int main(int argc, char** argv){
	network net;
	ncreate(net, INPUT_LAYER);
	nrand(net);

	char* test_labels;
	char** test_images;
	int num_test_images = read_MNIST(TEST_IMAGES, TEST_LABELS, test_images, test_labels);
	
	nsave(net);

	delete[] test_labels;
	for(int i = 0; i < num_test_images; i++){
		delete[] test_images[i];
	}
	delete[] test_images;
	
	nfree(net);
	nfree(net2);
        return 0;
}

