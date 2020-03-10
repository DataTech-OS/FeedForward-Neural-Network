#ifndef MNIST
#define MNIST

#include "include.hpp"

typedef struct mnist{
	vector<vector<double>> train_images;
	vector<vector<double>> train_labels;
	vector<vector<double>> test_images;
	vector<vector<double>> test_labels;
}mnist;

mnist *read_mnist_images();

#endif
