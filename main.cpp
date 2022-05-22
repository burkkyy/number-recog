#include <iostream>
#include <fstream>

#define P(x) std::cout << x << std::endl

#define IMAGES "data/train-images-idx3-ubyte"
#define LABELS "data/train-labels-idx1-ubyte"

#define INPUT_LAYER 784
#define Z1_LAYER 12
#define Z2_LAYER 12
#define OUTPUT_LAYER 10

float sigmoid(float x){
	return 0.5f * (x / (1 + abs(x)) + 1);
}

uint32_t swap_endian(uint32_t val){
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

void transfer(float* W, float* B, float* X, float* A, int num_W, int num_A){
	for(int i = 0; i < num_A; i++){
		A[i] = 0;
		for(int j = 0; j < num_W; j++){
			A[i] += W[j] * X[j]; 
		}
		A[i] = sigmoid(A[i] + B[i]);
	}
}

void forward_prop(float* in_layer, float* A1, float* W1, float* B1, float* A2, float* W2, float* B2, float* out_layer, float* W3, float* B3){
	transfer(W1, B1, in_layer, A1, INPUT_LAYER, Z1_LAYER);
	transfer(W2, B2, A1, A2, Z1_LAYER, Z2_LAYER);
	transfer(W3, B3, A2, out_layer, Z2_LAYER, OUTPUT_LAYER);
}

void init_net(float* W1, float* B1, float* W2, float* B2, float* W3, float* B3){
	// Randomly init weights
	for(int i = 0; i < INPUT_LAYER * Z1_LAYER; i++){
		W1[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < Z1_LAYER * Z2_LAYER; i++){
		W2[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < Z2_LAYER * OUTPUT_LAYER; i++){
		W3[i] = (float)rand() / RAND_MAX * 100;
	}

	// Randomly init biases
	for(int i = 0; i < Z1_LAYER; i++){
		B1[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < Z2_LAYER; i++){
		B2[i] = (float)rand() / RAND_MAX * 100;
	}

	for(int i = 0; i < OUTPUT_LAYER; i++){
		B3[i] = (float)rand() / RAND_MAX * 100;	
	}
}

int main(int argc, char** argv){
	std::ifstream images(IMAGES, std::ios::in | std::ios::binary);
	std::ifstream labels(LABELS, std::ios::in | std::ios::binary);
	
	uint32_t magic;
	uint32_t num_images;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	images.read((char*)&magic, sizeof(magic));
	magic = swap_endian(magic);
	if(magic != 2051){
		P("incorrect images file given");	
		exit(1);
	} else {
		P("correct images file given");
	}

	labels.read((char*)&magic, sizeof(magic));
	magic = swap_endian(magic);
	if(magic != 2049){
		P("incorrect lables file given");
		exit(1);
	} else {
		P("correct lables file given");
	}
	
	images.read((char*)&num_images, sizeof(num_images));
	num_images = swap_endian(num_images);
	P("Number of images: " << num_images);
	
	images.read((char*)&rows, sizeof(rows));
	rows = swap_endian(rows);
	P("Number of rows: " << rows);
	
	images.read((char*)&cols, sizeof(cols));
	cols = swap_endian(cols);
	P("Number of cols: " << cols);
	
	labels.read((char*)&num_labels, sizeof(num_labels));
	num_labels = swap_endian(num_labels);
	P("Number of labels: " << num_labels);
	
	char* pixels = new char[rows * cols];
	images.read(pixels, rows * cols);
	
	/*
	char label;
	for(int i = 0; i < 10; i++){
		labels.read(&label, sizeof(label));
		std::cout << (int)label << " ";	
	}
	std::cout << std::endl;
	*/

	images.close();
	labels.close();
	
	// input layer
	float* in_layer = new float[INPUT_LAYER]();
	
	// first hidden layer
	float* A1 = new float[Z1_LAYER]();
	float* W1 = new float[INPUT_LAYER * Z1_LAYER]();
	float* B1 = new float[Z1_LAYER]();
	
	// second hidden layer
	float* A2 = new float[Z2_LAYER]();
	float* W2 = new float[Z1_LAYER * Z2_LAYER]();
	float* B2 = new float[Z2_LAYER]();
	
	// output layer
	float* out_layer = new float[OUTPUT_LAYER]();	
	float* W3 = new float[Z2_LAYER * OUTPUT_LAYER]();
	float* B3 = new float[OUTPUT_LAYER]();	
	
	float* Y = new float[10];

	// set random values for the weights and biases
	init_net(W1, B1, W2, B2, W3, B3);

	for(int i = 0; i < INPUT_LAYER; i++){
		in_layer[i] = (float)pixels[i];
	}

	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			std::cout << in_layer[i * 28 + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	forward_prop(in_layer, A1, W1, B1, A2, W2, B2, out_layer, W3, B3);

	for(int i = 0; i < 10; i++){
		P("OUTPUT [" << i << "]: " << out_layer[i]);
	}

	/*
	for(int i = 0; i < 100; i++){
		std::cout << (float)rand() / RAND_MAX * 100 << std::endl;
	}
	*/
	
	delete[] in_layer;
	delete[] A1;
	delete[] W1;
	delete[] B1;
	delete[] A2;
	delete[] W2;
	delete[] B2;
	delete[] out_layer;
	delete[] W3;
	delete[] B3;
	delete[] Y;
	delete[] pixels;
	return 0;
}

