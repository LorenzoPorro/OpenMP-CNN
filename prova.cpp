#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;



/* convolution function
   input: input matrix, kernel matrix, output matrix, size of input, kernel and output matrices and the bias */
void convolution(int** input, float** kernel, float** output, int isize, int ksize, int osize, int bias){
    
    
    int stride=(isize - (int)(isize/ksize) * ksize);
    int iy,ix;  // input y and x coordinates

    for(int oy=0;oy<osize;oy++){

        iy=oy*stride;
        for(int ox=0;ox<osize;ox++){
            
            ix=ox*stride;
            for(int ky=0;ky<ksize;ky++){

                for(int kx=0;kx<ksize;kx++){

                    output[oy][ox] += input[iy][ix] * kernel[ky][kx] + bias;
                    ix++;
                }
                ix=ox*stride;
                iy++;
            }
            iy=oy*stride;
        }
    }

}



/* ReLU non-linear function: apply the ReLU nonlinear function f(x)=max(0,x) on each input value
   input: matrix on which applying ReLU function*/
void relu(float** relu, int size){

    for(int i=0;i<size;i++){

        for(int j=0;j<size;j++){

            relu[i][j]=max((float)0, relu[i][j]);
        }
    }
}



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, the size of both them, the size of the pooling matrix and the stride */
void maxpooling(int** input, int** output, int isize, int osize, int p, int stride){    // overlapping if: stride < p
    
    for(int oy=0;oy<osize;oy++){

        for(int ox=0;ox<osize;ox++){

            for(int fy=oy*stride;fy<p*stride;fy++){

                for(int fx=ox*stride;fx<p*stride;fx++){

                    output[oy][ox] = max(output[oy][ox], input[fy][fx]);
                }
            }
        }
    }

}





int main() {


    // input images divided in 3 layers
    int **input1;
    int **input2;
    int **input3;

    input1=new int *[224];
    for(int i=0;i<224;i++){
        input1[i]=new int [224];
    }
    input2=new int *[224];
    for(int i=0;i<224;i++){
        input2[i]=new int [224];
    }
    input3=new int *[224];
    for(int i=0;i<224;i++){
        input3[i]=new int [224];
    }

    // random bit values for each image layer
    for(int i=0;i<224;i++){
        for(int j=0;j<224;j++){
            input1[i][j]=rand()%10;
            input2[i][j]=rand()%10;
            input3[i][j]=rand()%10;
        }
    }


    // 3 kernels which will convolved on the 3 image matrices
    float **kernel1;
    float **kernel2;
    float **kernel3;
    
    kernel1=new float *[11];
    for(int i=0;i<11;i++){
        kernel1[i]=new float [11];
    }
    kernel2=new float *[11];
    for(int i=0;i<11;i++){
        kernel2[i]=new float [11];
    }
    kernel3=new float *[11];
    for(int i=0;i<11;i++){
        kernel3[i]=new float [11];
    }

    // random values between 0 and 1 for each kernel
    for(int i=0;i<11;i++){
        for(int j=0;j<11;j++){
            kernel1[i][j]=rand()%5 -2;
            kernel2[i][j]=rand()%5 -2;
            kernel3[i][j]=rand()%5 -2;
        }
    }


    // output of convolutional operation between input and kernel
    float **conv1;
    float **conv2;
    float **conv3;

    conv1=new float *[55];
    for(int i=0;i<55;i++){
        conv1[i]=new float [55];
    }
    conv2=new float *[55];
    for(int i=0;i<55;i++){
        conv2[i]=new float [55];
    }
    conv3=new float *[55];
    for(int i=0;i<55;i++){
        conv3[i]=new float [55];
    }
    
    // zero values in conv output matrix
    for(int i=0;i<55;i++){
        for(int j=0;j<55;j++){
            conv1[i][j]=0;
            conv2[i][j]=0;
            conv3[i][j]=0;
        }
    }


    // convolution between input image layers and kernels
    convolution(input1,kernel1,conv1,224,11,55,1);
    convolution(input2,kernel2,conv2,224,11,55,1);
    convolution(input3,kernel3,conv3,224,11,55,1);
    
    // ReLU nonlinearity
    relu(conv1,55);
    relu(conv2,55);
    relu(conv3,55);
}