#ifndef READ_MNSIT_H
#define READ_MNIST_H

#include <iostream>
#include <fstream>

uint32_t swap_endian(uint32_t);
int read_MNIST(const char*, const char*, char**&, char*&);

#endif

