#include "../include/include.hpp"
#include "../include/mnist.hpp"

mnist *read_mnist_images()
{
	//read MNIST training files into a buffer
	ifstream training_images, training_labels, test_images, test_labels;

	training_images.open("../data/train_images", ios::binary | ios::ate);
	training_labels.open("../data/train_labels", ios::binary | ios::ate);
	test_images.open("../data/test_images", ios::binary | ios::ate);
	test_labels.open("../data/test_labels", ios::binary | ios::ate);

	streamsize train_images_size = training_images.tellg(), 
			train_labels_size = training_labels.tellg(),
			test_images_size = test_images.tellg(), 
			test_labels_size = test_labels.tellg();

	training_images.seekg(0, ios::beg);	
	training_labels.seekg(0, ios::beg);
	test_images.seekg(0, ios::beg);	
	test_labels.seekg(0, ios::beg);

	vector<char> train_images_buffer(train_images_size), 
			train_labels_buffer(train_labels_size),
			test_images_buffer(test_images_size), 
			test_labels_buffer(test_labels_size);

	training_images.read(train_images_buffer.data(), train_images_size);
	training_labels.read(train_labels_buffer.data(), train_labels_size);
	test_images.read(test_images_buffer.data(), test_images_size);
	test_labels.read(test_labels_buffer.data(), test_labels_size);
	
	mnist *m = new(mnist);
	vector<double> image_place_holder(28*28);
	vector<double> label_place_holder(10);

	// training images
	for(int i = 0; i < 1000; i++)
	{
		for(int j = 0; j < 28*28; j++)
			image_place_holder[j] = (double)((unsigned char)train_images_buffer[16 + (i * 28*28) + j]) / 255;
		m->train_images.push_back(image_place_holder);
	}
	// test images
	for(int i = 0; i < 1000; i++)
	{
		for(int j = 0; j < 28*28; j++)
			image_place_holder[j] = (double)((unsigned char)test_images_buffer[16 + (i * 28*28) + j]) / 255;
		
		m->test_images.push_back(image_place_holder);
	}
	// train label
	for(int i = 0; i < 1000; i++)
	{
		for(int j = 0; j < 10; j++)
			j == (int)((unsigned char)train_labels_buffer[8 + i]) ? label_place_holder[j] = 1 : label_place_holder[j] = 0;
		
		m->train_labels.push_back(label_place_holder);
	}
	// test labels
	for(int i = 0; i < 1000; i++)
	{
		for(int j = 0; j < 10; j++)
			j == (int)((unsigned char)test_labels_buffer[8 + i]) ? label_place_holder[j] = 1 : label_place_holder[j] = 0;

		m->test_labels.push_back(label_place_holder);
	}

	return m;
}
