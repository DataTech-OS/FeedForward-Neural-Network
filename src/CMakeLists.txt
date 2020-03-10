cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

project(MNIST_feedforward_neural_network)

add_executable(
	mnist
	main.cpp
	mnist.cpp
	network.cpp
)
