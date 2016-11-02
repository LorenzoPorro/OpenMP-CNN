

class PoolingLayer{
    const int COLOR_DEPTH=3;
    int features;
    int height;
    int width;
    int depth;
    vector<vector<vector<int>>> input;
    vector<vector<vector<vector<double>>>> weights;
    vector<vector<vector<vector<double>>>> biases;

    PoolingLayer(vector<vector<vector<int>>> input, int features, int height, int width);

    vector<vector<vector<double>>> poolingOutput();

    vector<vector<vector<double>>> relu();



    std::default_random_engine gen;
    std::normal_distribution<float> distr(0,0.01);
};

class ConvLayer : public PoolingLayer{

    vector<vector<vector<vector<double>>>> kernel;

    ConvLayer(vector<vector<vector<int>>> input, int features, int height, int width);

    vector<vector<vector<double>>> convolution();

    vector<vector<double>> convOutput();
};