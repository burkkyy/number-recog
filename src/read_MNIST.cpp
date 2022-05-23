#include <iostream>
#include <fstream>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int read_MNIST(const char* images_filename, const char* labels_filename, char**& loaded_images, char*& loaded_labels){
	// Open files streams for images and labels file
	std::ifstream images(images_filename, std::ios::in | std::ios::binary);
	std::ifstream labels(labels_filename, std::ios::in | std::ios::binary);
	
	if(!images.is_open() || !labels.is_open()){
		std::cout << "error with filenames given" << std::endl;
		images.close();
		labels.close();
		return -1;
	}

	uint32_t magic;
	uint32_t num_images;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	images.read((char*)&magic, 4);
	magic = swap_endian(magic);
	if(magic != 2051){
		std::cout << "incorrect image file given" << std::endl;
		images.close();
		labels.close();
		return -1;
	}

	labels.read((char*)&magic, 4);
	magic = swap_endian(magic);
	if(magic != 2049){
		std::cout << "incorrect label file given" << std::endl;
		images.close();
		labels.close();
		return -1;
	}
	
	images.read((char*)&num_images, 4);
	num_images = swap_endian(num_images);
	
	labels.read((char*)&num_labels, 4);
	num_labels = swap_endian(num_labels);
	
	if(num_images != num_labels){
		std::cout << "number of images does not equal number of labels" << std::endl;
		images.close();
		labels.close();
		return -1;
	}

	images.read((char*)&rows, 4);
	rows = swap_endian(rows);
	
	images.read((char*)&cols, 4);
	cols = swap_endian(cols);
	
	if(rows * cols != 784){
		std::cout << "incorrect row or col size" << std::endl;
		images.close();
		labels.close();
		return -1;
	}

	loaded_labels = new char[num_labels];
	loaded_images = new char*[num_images];
	for(int i = 0; i < num_images; i++){
		loaded_images[i] = new char[rows * cols];
	}
	
	for (int i = 0; i < num_images; i++) {
    		images.read(loaded_images[i], rows * cols);
		labels.read(&loaded_labels[i], 1);
	}
	
	images.close();
	labels.close();
	return num_images;
	/* **mem clean up** for future reference :)
	delete[] loaded_labels;
	for(int i = 0; i < num_images; i++){
		delete[] loaded_images[i];
	}
	delete[] loaded_images;
	*/
}

