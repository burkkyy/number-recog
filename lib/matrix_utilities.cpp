#include "matrix.hpp"

mat* mcreate(int row, int col){
	mat* matrix = new mat;
	matrix->rows = row;
	matrix->cols = col;
	
	matrix->elements = new double*[row];
	for(int i = 0; i < row; i++){ matrix->elements[i] = new double[col](); }
	return matrix;
}

void mfree(mat* m){
	for(int i = 0; i < m->rows; i++){
		delete[] m->elements[i];
	}
	delete[] m;
	m = nullptr;
}

mat* mcopy(mat*& m){
	mat* matrix = mcreate(m->rows, m->cols);

	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			matrix->elements[i][j] = m->elements[i][j];
		}
	}

	return matrix;
}

void mmove(mat*& des, mat*& from){
	if(mcheck(des, from)){
		for(int i = 0; i < from->rows; i++){
			for(int j = 0; j < from->cols; j++){
				des->elements[i][j] = from->elements[i][j];
			}
		}
	} else {
		std::cout << "Dimensions do not align" << std::endl;
	}
}

void mprint(mat*& m){
	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			std::cout << m->elements[i][j] << " ";	
		}
		std::cout << std::endl;
	}
}

void mfill(mat*& m, int n){
	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			m->elements[i][j] = n;
		}
	}
}

void mrand(mat*& m){
	// set rand() seed
	//srand(NULL);
	srand(RAND_SEED);

	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			m->elements[i][j] = (double)(rand() % (2 * RAND_RANGE * RAND_SCALE) / RAND_SCALE) - RAND_RANGE;
		}
	}
}

void msave(char* filename, mat*& m){
	std::ofstream file(filename, std::ios::out);
	
	file << m->rows;
	file << m->cols;

	for(int i = 0; i < m->rows; i++){
		for(int j; j < m->cols; j++){
			file << m->elements[i][j];
		}
	}
	
	file.close();
}

mat* mload(char* filename){
	std::ifstream file(filename, std::ios::in);
	
	int rows, cols;
	file >> rows;
	file >> cols;
	
	mat* m = mcreate(rows, cols);
	double temp = 0.0f;

	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			file >> temp;
			m->elements[i][j] = temp;
		}
	}

	file.close();
	return m;
}

