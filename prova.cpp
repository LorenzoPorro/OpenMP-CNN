#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;


/* convolution function
   input: input matrix, kernel matrix, output matrix, the stride and the bias */
vector<vector<vector<float>>> convolution(vector<vector<vector<int>>> input, vector<vector<vector<vector<float>>>> kernel, vector<vector<vector<float>>> output, int stride, int bias){
    
    int iy,ix;  // input y and x coordinates
    int feat=output.size();
    int depth=input.size();
    int isize=input[0].size();
    int ksize=kernel[0].size();
    int osize=output[0].size();

    for(int f=0;f<feat;f++){    // for each feature

        for(int i=0;i<depth;i++){   // for each depth level

            // 2d-convolution
            for(int oy=0;oy<osize;oy++){

                iy=oy*stride;
                for(int ox=0;ox<osize;ox++){
            
                    ix=ox*stride;
                    for(int ky=0;ky<ksize;ky++){

                        for(int kx=0;kx<ksize;kx++){

                            output[f][oy][ox] += input[i][iy][ix] * kernel[f][i][ky][kx] + bias;
                            ix++;
                        }
                        ix=ox*stride;
                        iy++;
                    }
                    iy=oy*stride;
                }
            }
        }
    }

    return output;
}



/* ReLU non-linear function: apply the ReLU nonlinear function f(x)=max(0,x) to each input value
   input: matrix on which applying ReLU function */
vector<vector<vector<float>>> relu(vector<vector<vector<float>>> relu){

    int feat=relu.size();
    int size=relu[0].size();

    for(int f=0;f<feat;f++){

        for(int i=0;i<size;i++){

            for(int j=0;j<size;j++){
                
                relu[f][i][j]=max((float)0, relu[f][i][j]);
            }
        }
    }

    return relu;   
}



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, dimension of the overlapping pooling matrix and the stride */
void maxpooling(vector<vector<vector<int>>> input, vector<vector<vector<int>>> output, int p, int stride){    // overlapping if: stride < p
    
    int feat=input.size();
    int isize=input[0].size();
    int osize=output[0].size();

    for(int f=0;f<feat;f++){

        for(int oy=0;oy<osize;oy++){

            for(int ox=0;ox<osize;ox++){

                for(int fy=oy*stride;fy<p*stride;fy++){

                    for(int fx=ox*stride;fx<p*stride;fx++){

                        output[f][oy][ox] = max(output[f][oy][ox], input[f][fy][fx]);
                    }
                }
            }
        }
    }

}





int main() {


    // input images divided in 3 layers (224x224x3)
    vector<vector<vector<int>>> input(3, vector<vector<int>>(224, vector<int>(224)));
    
    // random bit values for each image layer
    for(int i=0;i<3;i++){

        for(int j=0;j<224;j++){

            for(int k=0;k<224;k++){

                input[i][j][k]=rand()%10;
            }
        }
    }



    // LAYER 1

    // number of features
    const int feat1=48;

    // 3 kernels which will convolved on the 3 image matrices (11x11x3xfeat)
    vector<vector<vector<vector<float>>>> kernel(feat, vector<vector<vector<float>>>(3, vector<vector<float>>(11, vector<float>(11))));
    
    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat;f++){
        for(int i=0;i<3;i++){
            for(int j=0;j<11;j++){
                for(int k=0;k<11;k++){
                    
                    kernel[f][i][j][k]=rand()%5 - 2;
                }
            }
        }
    }


    // output of convolutional operation between input and kernel of layer 1 (55x55xfeat)
    vector<vector<vector<float>>> layer1(feat, vector<vector<float>>(55, vector<float>(55)));
    
    // inizialize convolutional matrix with only zero values
    for(int f=0;f<feat;f++){
        for(int j=0;j<55;j++){
            for(int k=0;k<55;k++){

                layer1[f][j][k]=0;
            }
        }
    }


    int stride = (input[0].size() - (int)(input[0].size()/kernel[0].size()) * kernel[0].size());

    // convolution between input image layers and kernels
    layer1 = convolution(input,kernel,layer1,stride,1);

    // ReLU nonlinearity
    layer1 = relu(layer1);
    

    // LAYER 2



    // convolution between layer 1 and kernels

    // ReLU nonlinearity

    // overlapped max-pooling

}