#include <layer.hpp>

/*
* PoolingLayer constructor
*/
PoolingLayer::PoolingLayer(vector<vector<vector<int>>> input, int features, int height, int width){
    this.features = features;
    this.height = height;
    this.width = width;
    this.input = input;
    for(int f=0;f<features;f++){
        for(int h=0;h<height;h++){
            for(int w=0;j<width;w++){
                for(int c=0;k<COLOR_DEPTH;c++){       
                    this.weights[f][i][j][k]=distr(gen);
                }
            }
        }
    }
}

PoolingLayer::vector<vector<vector<double>>> poolingOutput(){
        //maxpooling
}

/*
* ConvLayer constructor
*/
ConvLayer::ConvLayer(vector<vector<vector<int>>> input, int features, int height, int width){
    this.features = features;
    this.height = height;
    this.width = width;
    this.input = input;
    for(int f=0;f<features;f++){
        for(int h=0;h<height;h++){
            for(int w=0;j<width;w++){
                for(int c=0;k<COLOR_DEPTH;c++){       
                    this.kernel[f][i][j][k]=distr(gen);
                }
            }
        }
    }
}

ConvLayer::vector<vector<vector<double>>> convolution(){
    //convolution
}

ConvLayer::vector<vector<double>> convOutput(){
        //conv+pooling
}