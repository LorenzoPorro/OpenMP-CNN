#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;

/*  1. filter dimension in max-pooling
    2. passing random dimension matrix to convolution function
    3. determine the number of neurons -> size of hidden layers
    4. apply relu */

/* convolution function
   input: input matrix, kernel matrix, output matrix, size of input, kernel and output matrices and the bias */
void convolution(int input[224][224], float kernel[11][11], float output[55][55], int bias){
    

    // determine the size of input, kernel and output rows and columns
    int isize=sizeof(input[0])/sizeof(int);
    int ksize=sizeof(kernel[0])/sizeof(int);
    int osize=sizeof(output[0])/sizeof(int);
    
    int stride=(isize - (int)(isize/ksize) * ksize);
    int iy,ix;  // input y and x

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



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, the size of both them, the size of the pooling matrix and the stride */
void maxpooling(int input[28][28], int output[14][14], int p, int stride){    // overlapping if: stride < p
    
    // determine the size of input and output rows and columns
    int isize=sizeof(input[0])/sizeof(int);
    int osize=sizeof(output[0])/sizeof(int);
    
    for(int oy=0;oy<osize;oy++){

        for(int ox=0;ox<osize;ox++){

            for(int fy=oy*stride;fy<p*stride;fy++){

                for(int fx=ox*stride;fx<p*stride;fx++){

                    output[oy][ox]=max(output[oy][ox], input[fy][fx]);
                }
            }
        }
    }

}



int main() {
    
    // input images divided in 3 layers
    int input1[224][224];
    int input2[224][224];
    int input3[224][224];

    // 3 kernels which will convolved on the 3 image matrices
    float kernel1[11][11];
    float kernel2[11][11];
    float kernel3[11][11];
    // bias
    int bias = 1;

    // output of convolutional operation between input and kernel
    float conv1[55][55];
    float conv2[55][55];
    float conv3[55][55];


    // random bit values for each image layer
    for(int i=0;i<224;i++){
        for(int j=0;j<224;j++){
            input1[i][j]=rand()%10;
            input2[i][j]=rand()%10;
            input3[i][j]=rand()%10;
        }
    }

    // random values between 0 and 1 for each kernel
    for(int i=0;i<11;i++){
        for(int j=0;j<11;j++){
            kernel1[i][j]=rand()%2;
            kernel2[i][j]=rand()%2;
            kernel3[i][j]=rand()%2;
        }
    }

    // zero values in conv output matrix
    for(int i=0;i<55;i++){
        for(int j=0;j<55;j++){
            conv1[i][j]=0;
            conv2[i][j]=0;
            conv3[i][j]=0;
        }
    }


    // convolution between input image layers matrices and kernels
    convolution(input1,kernel1,conv1,bias);
    convolution(input2,kernel2,conv2,bias);
    convolution(input3,kernel3,conv3,bias);
}