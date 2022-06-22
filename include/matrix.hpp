#ifndef MAT_H
#define MAT_H

#include <iostream>
#include <fstream>

//#define RAND_RANGE 3
//#define RAND_SCALE 10
#define RAND_SEED 7

struct mat {
	int rows;
	int cols;
	double** elements;
};

// matrix utilitys
bool mcheck(mat*&, mat*&);
mat* mcreate(int, int);
void mfree(mat*);
mat* mcopy(mat*&);
void mmove(mat*&, mat*&);

void mprint(mat*&);

void mfill(mat*&, int);
void mrand(mat*&);

void msave(char*, mat*&);
mat* mload(char*);

// matrix operations
mat* multiply(mat*&, mat*&);
mat* add(mat*&, mat*&);
mat* sub(mat*&, mat*&);
mat* dot(mat*&, mat*&);
mat* scale(double, mat*&);
mat* add_scalar(double, mat*&);
mat* transpose(mat*&);

void multiply(mat*&, mat*&, mat*&);
void add(mat*&, mat*&, mat*&);
void sub(mat*&, mat*&, mat*&);
void dot(mat*&, mat*&, mat*&);
void scale(double, mat*&, mat*&);
void transpose(mat*&, mat*&, mat*&);
void apply_func(double(*)(double), mat*, mat*&);

// apply_func RUTHLESSLY copied from:
// https://github.com/markkraay/mnist-from-scratch/blob/master/matrix/ops.c
mat* apply_func(double(*)(double), mat*);

#endif

