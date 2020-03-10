#ifndef NETWORK
#define NETWORK

#include "../include/include.hpp"

typedef struct layer_result{
	vector<vector<double>> y;
	vector<vector<double>> x;
}LR;

class network
{
public:	
	network(vector<int> topology);
	vector<vector<double>> feed_forward(vector<vector<double>>& input);
	void backprop(vector<vector<double>>& x, vector<vector<double>>& y);
private:
	vector<vector<double>> wx_b(vector<vector<double>>& x, int index);
	void sigmoid(vector<vector<double>>& x);
	double _sigmoid(double x);
	double sigmoid_prime(double x);
	int layers;
	vector<LR *> lr;
	vector<vector<vector<double>>> weights;
	vector<vector<vector<double>>> starting_weights;
	vector<vector<double>> biases; 
	vector<vector<double>> starting_biases;
};

#endif
