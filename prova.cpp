#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;

/*  1. filter dimension in max-pooling
    2. passing random dimension matrix to convolution function
    3. determine the number of neurons -> size of hidden layers */

// convolution function: takes in input the input matrix, the kernel matrix and the output
void convolution(int input[224][224], float kernel[11][11], float layer_conv[224][224], int size, int ksize, int bias){
    
    int ki,kj=(size - (int)(size/ksize) * ksize) / 2;

    for(int i=0;i<size;i++){

        for(int j=0;j<size;j++){

            layer_conv[i][j]=input[i][j]*kernel[ki][kj]+bias;

            kj++;
            if(kj>=ksize)
                kj=0;
        }

        kj=2;
        ki++;
        if(ki>=ksize)
            ki=0;
    }
}


// overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
void maxpooling(int conv[224][224], int layer2[48][48], int csize, int size){
    //filter dimension ???
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
    float conv1[224][224];
    float conv2[224][224];
    float conv3[224][224];


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


    // convolution between input image layers matrices and kernels
    convolution(input1,kernel1,conv1,224,11,bias);
    convolution(input2,kernel2,conv2,224,11,bias);
    convolution(input3,kernel3,conv3,224,11,bias);


}