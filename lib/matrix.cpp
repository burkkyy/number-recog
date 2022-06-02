#include "matrix.hpp"

bool mcheck(mat*& a, mat*& b){
	if(a->rows == b->rows && a->cols == b->cols){
		return true;
	}
	return false;
}

mat* multiply(mat*& a, mat*& b){
	if(mcheck(a, b)){
		mat* m = mcreate(a->rows, a->cols);
		for(int i = 0; i < a->rows; i++){
			for(int j = 0; j < b->cols; j++){
				m->elements[i][j] = a->elements[i][j] * b->elements[i][j];
			}
		}
		return m;
	} else {
		std::cout << "Dimensions do not align" << std::endl;
	}
	return nullptr;
}

mat* add(mat*& a, mat*& b){
	if(mcheck(a, b)){
		mat* m = mcreate(a->rows, a->cols);
		for(int i = 0; i < a->rows; i++){
			for(int j = 0; j < b->cols; j++){
				m->elements[i][j] = a->elements[i][j] + b->elements[i][j];
			}
		}
		return m;
	} else {
		std::cout << "Dimensions do not align" << std::endl;
	}
	return nullptr;
}

mat* sub(mat*& a, mat*& b){
	if(mcheck(a, b)){
		mat* m = mcreate(a->rows, a->cols);
		for(int i = 0; i < a->rows; i++){
			for(int j = 0; j < b->cols; j++){
				m->elements[i][j] = a->elements[i][j] - b->elements[i][j];
			}
		}
		return m;
	} else {
		std::cout << "Dimensions do not align" << std::endl;
	}
	return nullptr;
}

mat* dot(mat*& a, mat*& b){
	if(a->cols == b->rows){
		mat* m = mcreate(a->rows, b->cols);
		double sum = 0;
		for(int i = 0; i < a->rows; i++){
			for(int j = 0; j < b->cols; j++){
				sum = 0;
				for(int k = 0; k < b->rows; k++){
					sum += a->elements[i][k] * b->elements[k][j];
				}
				m->elements[i][j] = sum;
			}
		}
		return m;
	} else {
		std::cout << "Dimensions do not align" << std::endl;
	}
	return nullptr;
}

mat* scale(double num, mat*& m){
	mat* matrix = mcopy(m);
	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			matrix->elements[i][j] *= num;
		}
	}
	return matrix;
}

mat* add_scale(double num, mat*& m){
	mat* matrix = mcopy(m);
	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			matrix->elements[i][j] += num;
		}
	}
	return matrix;
}

mat* transpose(mat*& m){
	mat* matrix = mcreate(m->cols, m->rows);
	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			matrix->elements[j][i] = m->elements[i][j];
		}
	}
	return matrix;
}

mat* apply_func(double(*func)(double), mat* m){
	mat* matrix = mcopy(m);
	for(int i = 0; i < m->rows; i++){
		for(int j = 0; j < m->cols; j++){
			matrix->elements[i][j] = (*func)(m->elements[i][j]);
		}
	}
	return matrix;
}

