#include "../include/include.hpp"
#include "../include/network.hpp"

network::network(vector<int> topology)
{
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<> dis{0,1};

	// how many layers the network contains
	this->layers = topology.size() - 1;
	// how many matrixes of weights the network has
	this->weights.resize(topology.size() - 1);
	this->starting_weights.resize(topology.size() - 1);
	
	// resize each layer to contain the appropriate number of weights
	for(int i = 0; i < topology.size() - 1; i++)
	{
		this->weights[i].resize(topology[i + 1]); // how many neurons
		this->starting_weights[i].resize(topology[i + 1]);
	}

	// dunno why two for loops are needed but otherwise the program segfaults (damn c++ internal)
	for(int i = 0; i < topology.size() - 1; i++)
	{
		for(int j = 0; j < topology[i + 1]; j++)
		{
			this->weights[i][j].resize(topology[i]); // how many weights
			this->starting_weights[i][j].resize(topology[i]);
		}
	}
	
	// create the biases
	this->biases.resize(topology.size() - 1);
	this->starting_biases.resize(topology.size() - 1);

	// resize each vector of biases according to the number of neurons of the corresponding layer
	for(int i = 0; i < topology.size() - 1; i++)
	{
		this->biases[i].resize(topology[i + 1]);
		this->starting_biases[i].resize(topology[i + 1]);
	}

	// time to fill the weights and the biases
	for(int i = 0; i < this->weights.size(); i++)
		for(int j = 0; j < this->weights[i].size(); j++)	
			for(int k = 0; k < this->weights[i][j].size(); k++)
				this->weights[i][j][k] = dis(gen);
    
	for(int i = 0; i < this->biases.size(); i++)	
		for(int j = 0; j < this->biases[i].size(); j++)
			this->biases[i][j] = dis(gen);

	// create the container structure
	for(int i = 0; i < this->layers; i++){
		LR *lr = new(LR);
		this->lr.push_back(lr);
	}	
}

vector<vector<double>> network::feed_forward(vector<vector<double>>& input)
{
	vector<vector<double>> x = input;

	for(int i = 0; i < this->layers; i++)
	{
		this->lr[i]->x = x;
		x = wx_b(x, i);
		this->lr[i]->y = x;
		sigmoid(x);
	}	

	return x;
}

vector<vector<double>> network::wx_b(vector<vector<double>>& x, int index)
{
	vector<vector<double>> res(x.size());

	if(x[0].size() == this->weights[index][0].size())
	{
		for(int i = 0; i < res.size(); i++)
			res[i].resize(this->weights[index].size(), 0);
		
		for(int i = 0; i < res.size(); i++)
			for(int j = 0; j < res[i].size(); j++)
				for(int k = 0; k < x[i].size(); k++)
					res[i][j] += x[i][k] * this->weights[index][j][k];		

		for(int i = 0; i < res.size(); i++)
			for(int j = 0; j < res[i].size(); j++)
				res[i][j] += this->biases[index][j];
	}
	else
	{
		cout << "Input and weights sizes do not match" << endl;
		exit(-1);
	}
	
	return res;
}

void network::sigmoid(vector<vector<double>>& x)
{
	for(int i = 0; i < x.size(); i++)
		for(int j = 0; j < x[i].size(); j++)
			x[i][j] = 1 / (1 + exp(-x[i][j]));
}

double network::_sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double network::sigmoid_prime(double x)
{
	double sigmoid = 1 / (1 + exp(-x));
	return sigmoid * (1 - sigmoid);
}

void network::backprop(vector<vector<double>>& x, vector<vector<double>>& y)
{
	vector<vector<double>> nabla_weights;
	vector<double> gradient(10);
	vector<double> next_grad(this->weights[this->layers - 1][0].size(), 0);

	gradient.resize(2);

	for(int i = 0; i < this->weights.size(); i++)
		for(int j = 0; j < this->weights[i].size(); j++)	
			for(int k = 0; k < this->weights[i][j].size(); k++)
				this->starting_weights[i][j][k] = this->weights[i][j][k];
    
	for(int i = 0; i < this->biases.size(); i++)	
		for(int j = 0; j < this->biases[i].size(); j++)
			this->starting_biases[i][j] = this->biases[i][j];

	// allocate weights vector with the proper sizes
	nabla_weights.resize(this->lr[this->layers - 1]->x[0].size());
	for(int i = 0; i < nabla_weights.size(); i++)
		nabla_weights[i].resize(gradient.size());

	// loop through each image and backpropagate
	for(int j = 0; j < x.size(); j++)
	{
		// set nabla weights to zero
		for(int t = 0; t < nabla_weights.size(); t++)
			for(int l = 0; l < nabla_weights[t].size(); l++)
				nabla_weights[t][l] = 0;
		
		next_grad.resize(this->weights[this->layers - 1][0].size(), 0);
		// set next_grad to zero
		for(int t = 0; t < next_grad.size(); t++)
			next_grad[t] = 0;

		// for layer 0 calculate y_hat - y (after the sigmoid activation)
		for(int i = 0; i < gradient.size(); i++)
			gradient[i] = this->_sigmoid(this->lr[this->layers - 1]->y[j][i]) - y[j][i];

		// compute the nabla for the weights
		for(int i = 0; i < nabla_weights.size(); i++)
			for(int k = 0; k < nabla_weights[k].size(); k++)
				nabla_weights[i][k] += gradient[k] * this->lr[this->layers - 1]->x[j][i];		

		// compute the gradient wrt the previous layer output
		for(int i = 0; i < next_grad.size(); i++)		
			for(int k = 0; k < this->starting_weights[this->layers - 1].size(); k++)
				next_grad[i] += this->starting_weights[this->layers - 1][k][i] * gradient[k];
	
		// adjust the weights of the first layer
		for(int i = 0; i < nabla_weights.size(); i++)
			for(int k = 0; k < nabla_weights[i].size(); k++)
				this->weights[this->layers - 1][k][i] -= 0.1 * nabla_weights[i][k];
		
		// adjust the biases
		for(int i = 0; i < this->biases[this->layers - 1].size(); i++)
			this->biases[this->layers - 1][i] -= 0.1 * gradient[i];
		
		for(int i = this->layers - 1; i > 0; i--)
		{
			// calculate the derivative wrt the output of XW' + b
			vector<double> g(next_grad.size());
			for(int k = 0; k < this->lr[i - 1]->y[j].size(); k++)
				g[k] = next_grad[k] * this->sigmoid_prime(this->lr[i - 1]->y[j][k]);

			vector<vector<double>> _nabla_weights;
			_nabla_weights.resize(this->lr[i - 1]->x[j].size());
			for(int i = 0; i < _nabla_weights.size(); i++)
			{
				_nabla_weights[i].clear();
				for(int k = 0; k < g.size(); k++)
					_nabla_weights[i].push_back(0);
			}

			// compute the nabla for the weights
			for(int t = 0; t < _nabla_weights.size(); t++)
				for(int k = 0; k < _nabla_weights[k].size(); k++)
					_nabla_weights[t][k] += g[k] * this->lr[i - 1]->x[j][t];			

			// compute the gradient wrt the previous layer output
			if(i != 1)
			{
				next_grad.resize(this->starting_weights[i - 1][0].size(), 0);
				for(int t = 0; t < this->starting_weights[i - 1][0].size(); t++)
					for(int k = 0; k < this->starting_weights[i - 1].size(); k++)	
						next_grad[t] += this->starting_weights[i - 1][k][t] * g[k];
			}

			// adjust the weights of the current layer
			for(int t = 0; t < _nabla_weights.size(); t++)
				for(int k = 0; k < _nabla_weights[t].size(); k++)
					this->weights[i - 1][k][t] -= 0.1 * _nabla_weights[t][k];
		
			// adjust the biases
			for(int t = 0; t < this->biases[i - 1].size(); t++)
				this->biases[i - 1][t] -= 0.1 * g[t];
		}			
	}
}
/*
	for(int i = 0; i < this->layers; i++)
	{
		cout << "Layer " << i + 1 << endl << endl << "X : " << endl;
		for(int j = 0; j < this->lr[i]->x.size(); j++)
		{
			for(int k = 0; k < this->lr[i]->x[j].size(); k++)
				cout << this->lr[i]->x[j][k] << " ";
			cout << endl;
		}
		
		cout << endl << "Y : " << endl;
		for(int j = 0; j < this->lr[i]->y.size(); j++)
		{
			for(int k = 0; k < this->lr[i]->y[j].size(); k++)
				cout << this->lr[i]->y[j][k] << " ";
			cout << endl;
		}
		cout << endl;
	}

	cout << "FINAL OUTPUT" << endl;
	for(int i = 0; i < x.size(); i++)
	{
		for(int j = 0; j < x[i].size(); j++)
			cout << x[i][j] << " ";
		cout << endl;
	}
*/
/*
		cout << "NABLA WEIGHTS" << endl << endl;
		for(int t = 0; t < nabla_weights.size(); t++)
		{
			for(int l = 0; l < nabla_weights[t].size(); l++)
				cout << nabla_weights[t][l] << " ";
			cout << endl;
		}

		cout << "PREVIOUS LAYER GRADIENT" << endl << endl;
		for(int t = 0; t < next_grad.size(); t++)
			cout << next_grad[t] << " ";
		cout << endl;
	
*/
/*
			cout << "NABLA WEIGHTS" << endl << endl;
			for(int t = 0; t < _nabla_weights.size(); t++)
			{
				for(int l = 0; l < _nabla_weights[t].size(); l++)
					cout << _nabla_weights[t][l] << " ";
				cout << endl;
			}
*/
/*
	for(int s = 0; s < next_grad.size(); s++)
				cout << next_grad[s] << " ";
			cout << endl << endl;
*/
