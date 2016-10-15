#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;

// THE NUMBER OF FEATURES IS 1



/* convolution function
   input: input matrix, kernel matrix, output matrix, size of input, kernel  and their depth and size of output, the bias and the number of features */
void convolution(int*** input, float**** kernel, float*** output, int isize, int ksize, int depth, int osize, int bias, int feat){
    
    int stride=(isize - (int)(isize/ksize) * ksize);
    int iy,ix;  // input y and x coordinates
    float acc[depth][osize][osize];  // output of the convolution between each layer of input and each layer of kernel (55x55x3)

    // initialization of the accumulator acc
    for(int i=0;i<depth;i++){
        for(int j=0;j<osize;j++){
            for(int k=0;k<osize;k++){
                acc[i][j][k]=0;
            }
        }
    }

    // convolution
    for(int f=0;f<feat;f++){

        for(int i=0;i<depth;i++){

            for(int oy=0;oy<osize;oy++){

                iy=oy*stride;
                for(int ox=0;ox<osize;ox++){
            
                    ix=ox*stride;
                    for(int ky=0;ky<ksize;ky++){

                        for(int kx=0;kx<ksize;kx++){

                            acc[i][oy][ox] += input[i][iy][ix] * kernel[f][i][ky][kx] + bias;
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

    // sum each element on same position to convert 55x55x3 acc into 55x55x1 output
    for(int f=0;f<feat;f++){
        for(int i=0;i<osize;i++){
            for(int j=0;j<osize;j++){
                for(int d=0;d<depth;d++){
                    output[f][i][j] += acc[d][i][j];
                }
            }
        }
    }

}



/* ReLU non-linear function: apply the ReLU nonlinear function f(x)=max(0,x) to each input value
   input: matrix on which applying ReLU function, size of matrix and number of features*/
void relu(float*** relu, int size,int feat){

    for(int f=0;f<feat;f++){

        for(int i=0;i<size;i++){

            for(int j=0;j<size;j++){

                relu[f][i][j]=max((float)0, relu[f][i][j]);
            }
        }
    }
        
}



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, the size of both them, the size of the pooling matrix and the stride */
void maxpooling(int*** input, int*** output, int isize, int idepth, int osize, int p, int stride){    // overlapping if: stride < p
    
    for(int i=0;i<idepth;i++){

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

    // number of features
    const int feat=1;

    // to pass matrices of different dimensions to a function we use a pointer of pointers
    // input images divided in 3 layers (224x224x3)
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


    // 3 kernels which will convolved on the 3 image matrices (11x11x3xfeat)
    float ****kernel;
    
    kernel=new float ***[feat];
    for(int i=0;i<3;i++){
        kernel[i]=new float **[3];

        for(int j=0;j<11;j++){
            kernel[i][j]=new float *[11];

            for(int k=0;k<11;k++){

                kernel[i][j][k]= new float [11];
            }
        }
    }

    
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
    float ***layer1;
    
    layer1=new float **[feat];
    for(int i=0;i<55;i++){
        layer1[i]=new float *[55];

        for(int j=0;j<55;j++){

            layer1[i][j]=new float [55];
        }
    }

    
    // inizialize convolutional matrix with only zero values
    for(int f=0;f<feat;f++){
        for(int j=0;j<55;j++){
            for(int k=0;k<55;k++){

                layer1[f][j][k]=0;
            }
        }
    }



    // LAYER 1
    // convolution between input image layers and kernels
    convolution(input,kernel,layer1,224,11,3,55,1,feat);
    
    // ReLU nonlinearity
    relu(layer1,55,feat);


    // LAYER 2
    // convolution between layer 1 and kernels

    // ReLU nonlinearity

    // overlapped max-pooling
    
     
}