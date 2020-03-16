#include "../include/include.hpp"
#include "../include/mnist.hpp"
#include "../include/network.hpp"

int main(int argc, char **argv)
{
	vector<int> topology = {784, 30, 10};
	
	mnist *m = read_mnist_images();

	network *nn = new network(topology);

	for(int i = 0; i < 100000; i++)
	{
		//if(i % 10 == 0)
		cout << i << endl;
		nn->feed_forward(m->train_images);
		nn->backprop(m->train_images, m->train_labels);
	}

	vector<vector<double>> x = nn->feed_forward(m->train_images);	

	cout << "FINAL OUTPUT" << endl;
	for(int i = 0; i < x.size(); i++)
	{
		for(int j = 0; j < x[i].size(); j++)
			cout << x[i][j] << " ";
		cout << endl;
	}

	return 0;
}	
