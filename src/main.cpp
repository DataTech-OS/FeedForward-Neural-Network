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

/*
	vector<vector<vector<double>>> a(2);
    int t = 0;
    
    for(int i = 0; i < 2; i++)
        a[i].resize(3);

	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 3; j++)
			a[i][j].resize(4);
    
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 4; k++){
                a[i][j][k] = t++;
                cout << a[i][j][k] << " ";
            }
            cout << endl;
        } 
        cout << endl << endl;
    }
*/
/*
	cout << "WEIGHTS" << endl << endl;
	for(int i = 0; i < this->weights.size(); i++){
		for(int j = 0; j < this->weights[i].size(); j++){
			for(int k = 0; k < this->weights[i][j].size(); k++)
				cout << this->weights[i][j][k] << " ";
    		cout << endl;
		}
		cout << endl << endl << endl;
	}
	
	cout << "BIASES" << endl << endl;
	for(int i = 0; i < this->biases.size(); i++){
		for(int j = 0; j < this->biases[i].size(); j++)
			cout << this->biases[i][j] << " ";
		cout << endl << endl;
	}
*/
/*
	cout << "RESULT" << endl;
	for(int i = 0; i < x.size(); i++){
		for(int j = 0; j < x[i].size(); j++)
			cout << x[i][j] << " ";
		cout << endl;
	}
	
	cout << "HERE" << endl << endl;
*/

/*
	int t = 7;
	vector<vector<double>> input(4);
	for(int i = 0; i < input.size(); i++)
		input[i].resize(5);
	
	cout << "INPUT" << " " << input.size() << " x " << input[0].size() << endl;
	for(int i = 0; i < input.size(); i++){
		for(int j = 0; j < input[i].size(); j++){
			input[i][j] = (double)t--;
			cout << input[i][j] << " ";
		}
		cout << endl;
		t = 3;
	}

	cout << endl << endl;
*/	
/*
	vector<vector<double>> test = {	
									{.43, -.21, .33, .18, -.2, .43, -.21, .33, .18, -.2}, 
									{-.33, -.004, -.12, .09, -.01, -.33, -.004, -.12, .09, -.01}, 
									{.001, .02, .077, -.98, -.3, .001, .02, .077, -.98, -.3} 
								};

	vector<vector<double>> y = {
								{0, 1},
								{1, 0},
								{0, 1}
							};
*/
/*
	vector<int> topology = {10, 9, 8, 7, 6, 5, 4, 3, 2};
	vector<vector<double>> test = {	
									{.43, -.21, .33, .18, -.2, .43, -.21, .33, .18, -.2}, 
									{-.33, -.004, -.12, .09, -.01, -.33, -.004, -.12, .09, -.01}, 
									{.001, .02, .077, -.98, -.3, .001, .02, .077, -.98, -.3} 
								};

	vector<vector<double>> y = {
								{0, 1},
								{1, 0},
								{0, 1}
							};

	mnist *m = read_mnist_images();

	network *nn = new network(topology);

	for(int i = 0; i < 10000; i++)
	{
		nn->feed_forward(test);
		nn->backprop(test, y);
	}

	nn->feed_forward(test);
*/		
