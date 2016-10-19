#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;


/*TODO
    - improve code like functions' types and outputs */

/* convolution function
   input: input matrix, kernel matrix, output matrix, the stride */
vector<vector<vector<float>>> convolution(vector<vector<vector<float>>> input, vector<vector<vector<vector<float>>>> kernel, vector<vector<vector<float>>> output, int stride){
    
    int bias=1;
    int iy,ix;  // input y and x coordinates
    int feat=output.size();
    int depth=input.size();
    int isize=input[0].size();
    int ksize=kernel[0][0].size();
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
                            
                            // check if the kernel goes outside the input matrix
                            if(ky*oy<isize && kx*ox<isize && iy<isize && ix<isize){
                                output[f][oy][ox] += input[i][iy][ix] * kernel[f][i][ky][kx] + bias;
                            }
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
vector<vector<vector<float>>> maxpooling(vector<vector<vector<float>>> input, vector<vector<vector<float>>> output){
    
    int p=3;    // max-pooling matrix size (3x3)
    int stride=2;   // overlapping: stride < p
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

    return output;
}





int main() {


    // input images divided in 3 layers (224x224x3)
    vector<vector<vector<float>>> input(3, vector<vector<float>>(224, vector<float>(224)));
    
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
    const int feat1=1;

    // 3 kernels which will convolved on the 3 image matrices (11x11x3xfeat1)
    vector<vector<vector<vector<float>>>> kernel1(feat1, vector<vector<vector<float>>>(3, vector<vector<float>>(11, vector<float>(11))));
    
    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat1;f++){
        for(int i=0;i<3;i++){
            for(int j=0;j<11;j++){
                for(int k=0;k<11;k++){
                    
                    kernel1[f][i][j][k]=rand()%5 - 2;
                }
            }
        }
    }


    // output of convolutional operation between input and kernel of layer 1 (55x55xfeat1)
    vector<vector<vector<float>>> layer1(feat1, vector<vector<float>>(55, vector<float>(55)));


    int stride1 = round((float)(input[0].size() - kernel1[0][0].size())/(layer1[0].size()-1));
    
    // convolution between input image layers and kernels
    layer1 = convolution(input,kernel1,layer1,stride1);
    
    // ReLU nonlinearity
    layer1 = relu(layer1);
    


    // LAYER 2

    // number of features
    const int feat2=1;

    // kernel (5x5xfeat1xfeat2)
    vector<vector<vector<vector<float>>>> kernel2(feat2, vector<vector<vector<float>>>(feat1, vector<vector<float>>(5, vector<float>(5))));
    
    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat2;f++){
        for(int i=0;i<feat1;i++){
            for(int j=0;j<5;j++){
                for(int k=0;k<5;k++){
                    
                    kernel2[f][i][j][k]=rand()%5 - 2;
                }
            }
        }
    }

    // output of convolutional operation between layer1 and kernel of layer 2 (55x55xfeat2)
    vector<vector<vector<float>>> conv2(feat2, vector<vector<float>>(55, vector<float>(55)));

    // output of overlapped max-pooling operation (27x27xfeat2)
    vector<vector<vector<float>>> layer2(feat2, vector<vector<float>>(27, vector<float>(27)));

    
    int stride2 = round((float)(layer1[0].size() - kernel2[0][0].size())/(conv2[0].size()-1));
    
    // convolution between layer 1 and kernels of layer 2
    conv2 = convolution(layer1,kernel2,conv2,stride2);
    
    // ReLU nonlinearity
    conv2 = relu(conv2);
    
    // overlapped max-pooling    
    layer2 = maxpooling(conv2,layer2);
    cout<<"end layer2 pool"<<endl;


    // LAYER 3

    
}