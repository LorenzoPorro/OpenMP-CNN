#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;





/* convolution function
   input: input matrix, kernel matrix, output matrix, size of input, kernel and output matrices and the bias */
void convolution(int*** input, float*** kernel, float*** output, int isize, int ksize, int osize, int bias){
    
    int stride=(isize - (int)(isize/ksize) * ksize);
    int iy,ix;  // input y and x coordinates

    for(int i=0;i<3;i++){

        for(int oy=0;oy<osize;oy++){

            iy=oy*stride;
            for(int ox=0;ox<osize;ox++){
            
                ix=ox*stride;
                for(int ky=0;ky<ksize;ky++){

                    for(int kx=0;kx<ksize;kx++){

                        output[i][oy][ox] += input[i][iy][ix] * kernel[i][ky][kx] + bias;
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



/* ReLU non-linear function: apply the ReLU nonlinear function f(x)=max(0,x) to each input value
   input: matrix on which applying ReLU function*/
void relu(float*** relu, int size){

    for(int k=0;k<3;k++){

        for(int i=0;i<size;i++){

            for(int j=0;j<size;j++){

                relu[k][i][j]=max((float)0, relu[k][i][j]);
            }
        }
    }
        
}



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, the size of both them, the size of the pooling matrix and the stride */
void maxpooling(int*** input, int*** output, int isize, int osize, int p, int stride){    // overlapping if: stride < p
    
    for(int i=0;i<3;i++){

        for(int oy=0;oy<osize;oy++){

            for(int ox=0;ox<osize;ox++){

                for(int fy=oy*stride;fy<p*stride;fy++){

                    for(int fx=ox*stride;fx<p*stride;fx++){

                        output[i][oy][ox] = max(output[i][oy][ox], input[i][fy][fx]);
                    }
                }
            }
        }
    }

}





int main() {


    // to pass matrices of different dimensions to a function we use a pointer of pointers
    // input images divided in 3 layers
    int ***input;

    input=new int **[3];
    for(int i=0;i<224;i++){
        input[i]=new int *[224];

        for(int j=0;j<224;j++){

            input[i][j]=new int [224];
        }
    }

    // random bit values for each image layer
    for(int i=0;i<3;i++){

        for(int j=0;j<224;j++){

            for(int k=0;k<224;k++){

                input[i][j][k]=rand()%10;
            }
        }
    }


    // 3 kernels which will convolved on the 3 image matrices
    float ***kernel;
    
    kernel=new float **[3];
    for(int i=0;i<11;i++){
        kernel[i]=new float *[11];

        for(int j=0;j<11;j++){

            kernel[i][j]=new float [11];
        }
    }

    
    // random values between -2 and 2 for each kernel
    for(int i=0;i<3;i++){

        for(int j=0;j<11;j++){

            for(int k=0;k<11;k++){

                kernel[i][j][k]=rand()%5 - 2;
            }
        }
    }


    // output of convolutional operation between input and kernel of layer 1
    float ***layer1;
    
    layer1=new float **[3];
    for(int i=0;i<55;i++){
        layer1[i]=new float *[55];

        for(int j=0;j<55;j++){

            layer1[i][j]=new float [55];
        }
    }

    
    // inizialize convolutional matrix with only zero values
    for(int i=0;i<3;i++){

        for(int j=0;j<55;j++){

            for(int k=0;k<55;k++){

                layer1[i][j][k]=0;
            }
        }
    }



    // LAYER 1
    // convolution between input image layers and kernels
    convolution(input,kernel,layer1,224,11,55,1);
    
    // ReLU nonlinearity
    relu(layer1,55);


    // LAYER 2
    // convolution between layer 1 and kernels

    // ReLU nonlinearity

    // overlapped max-pooling
    
     
}