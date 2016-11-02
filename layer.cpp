#include </home/lorenzo/OpenMP-CNN/layer.hpp>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

/*
* PoolingLayer constructor
*/
PoolingLayer::PoolingLayer(vector<double> &output){
    size = output.size();
    std::default_random_engine gen;
    std::normal_distribution<float> distr(0,0.01);
    for(int s=0;s<size;s++){
        cout << s << ", ";
        output.resize(output.size()); 
        output[s]=0;
        weights[s]=distr(gen);
        biases[s]=1;
    }
}

void PoolingLayer::layerMax(vector<double> input){
    for(int i=0;i<output.size();i++){
        for(int j=0;j<input.size();j++){
            output[i]=max(input[j], output[i]);
        }
    }
}

void PoolingLayer::layerMax(vector<vector<vector<double>>> input){
    for(int i=0;i<output.size();i++){
        for(int j=0;j<input.size();j++){
            for(int k=0;k<input[0].size();k++){
                for(int z=0;z<input[0][0].size();z++){
                    output[i]=max(output[i], input[j][k][z]);
                }
            }
        }
    }
}

/*
* ConvLayer constructor
*/
ConvLayer::ConvLayer(int layerNumber, vector<vector<vector<double>>> &convOut){
    features = convOut.size();
    height = convOut[0].size();
    width = convOut[0][0].size();
    for(int f=0;f<features;f++){
        for(int h=0;h<height;h++){
            for(int w=0;w<width;w++){
                output[f][h][w]=0;
                convOutput[f][h][w]=0;
                if(layerNumber==2 || layerNumber==4 || layerNumber==5)  biases[f][h][w]=1;
                else    biases[f][h][w]=0;
            }
        }
    }
    std::default_random_engine gen;
    std::normal_distribution<float> distr(0,0.01);
    for(int f=0;f<features;f++){
        for(int h=0;h<height;h++){
            for(int w=0;w<width;w++){
                for(int c=0;c<COLOR_DEPTH;c++){
                    kernel[f][c][h][w]=distr(gen);
                }
            }
        }
    }
}

void ConvLayer::relu(){

    int feat=convOutput.size();
    int size=convOutput[0].size();

    for(int f=0;f<feat;f++){

        for(int i=0;i<size;i++){

            for(int j=0;j<size;j++){
                
                convOutput[f][i][j]=max((double)0, convOutput[f][i][j]);
            }
        }
    }

}

/* convolution function
   input: input matrix, kernel matrix, output matrix, the stride and the bias */
void ConvLayer::convolution(vector<vector<vector<double>>> input, int stride){
    

    int iy,ix;  // input y and x coordinates
    int feat=convOutput.size();
    int depth=input.size();
    int isize=input[0].size();
    int ksize=kernel[0][0].size();
    int osize=convOutput[0].size();


    for(int f=0;f<feat;f++){    // for each feature
    
        for(int i=0;i<depth;i++){   // for each depth level
            
            // 2d-convolution
            for(int oy=0;oy<osize;oy++){
                
                iy=oy*stride;
                for(int ox=0;ox<osize;ox++){
                    
                    ix=ox*stride;
                    for(int ky=0;ky<ksize;ky++){
                        
                        for(int kx=0;kx<ksize;kx++){
                            
                            // check if the kernel goes outside the input matrix
                            if(ky*oy<isize && kx*ox<isize && iy<isize && ix<isize){
                                convOutput[f][oy][ox] += input[i][iy][ix] * kernel[f][i][ky][kx];
                            }
                            ix++;
                        }
                        ix=ox*stride;
                        iy++;
                    }
                    iy=oy*stride;
                    convOutput[f][oy][ox] += biases[f][oy][ox];
                }
            }
        }
    }
}

void ConvLayer::maxpooling(vector<vector<vector<double>>> maxInput){
    
    int p=3;    // max-pooling matrix size (3x3)
    int stride=2;   // overlapping: stride < p
    int feat=maxInput.size();
    int isize=maxInput[0].size();
    int osize=output[0].size();

    for(int f=0;f<feat;f++){

        for(int oy=0;oy<osize;oy++){

            for(int ox=0;ox<osize;ox++){

                for(int fy=oy*stride;fy<p*stride;fy++){

                    for(int fx=ox*stride;fx<p*stride;fx++){

                        output[f][oy][ox] = max(output[f][oy][ox], maxInput[f][fy][fx]);
                    }
                }
            }
        }
    }

}
