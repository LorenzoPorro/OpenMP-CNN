#include <vector>

using namespace std;

#ifndef LAYER_HPP
#define LAYER_HPP

class PoolingLayer{
    public:
    const int COLOR_DEPTH=3;
    int size;
    vector<double> weights;
    vector<double> biases;
    vector<double> output;
    PoolingLayer(vector<double> &output);
    void layerMax(vector<double> input);
    void layerMax(vector<vector<vector<double>>> input);
};

class ConvLayer{
    public:
    const int COLOR_DEPTH=3;
    int features;
    int height;
    int width;
    vector<vector<vector<double>>> output;
    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> biases;
    vector<vector<vector<vector<double>>>> kernel;
    vector<vector<vector<double>>> convOutput;

    ConvLayer(int layerNumber, vector<vector<vector<double>>> &convOut);
    void relu();
    void convolution(vector<vector<vector<double>>> input, int stride);
    void maxpooling(vector<vector<vector<double>>> maxInput);
};

#endif /* LAYER_HPP*/