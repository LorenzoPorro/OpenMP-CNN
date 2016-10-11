#include <cstdlib>
#include <omp.h>
#include <iostream>

using namespace std;


/*convolution function: takes in input the input matrix, the kernel matrix and the output*/
void convolution(int input[224][224], float kernel[11][11], float layer_conv[224][224], int size, int ksize){
    
    int kcenterx=ksize/2;
    int kcentery=ksize/2;

    for(int i=0;i<size;i++){

        for(int j=0;j<size;j++){

            for(int k=0;k<ksize;k++){

                int kk=ksize-1-k;;
                for(int l=0;l<ksize;l++){

                    int ll=ksize-1-l;

                    int ii=i+(k-kcentery);
                    int jj=j+(l-kcenterx);

                    if(ii>=0 && ii<size && jj>=0 && jj<size){
                        layer_conv[i][j]+=input[ii][jj]*kernel[kk][ll];
                    }
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

    // random bit values for each image layer
    for(int i=0;i<224;i++){
        for(int j=0;j<224;j++){
            input1[i][j]=rand()%10;
            input2[i][j]=rand()%10;
            input3[i][j]=rand()%10;
        }
    }

    // 3 kernels which will convolved on the 3 image matrices
    float kernel1[11][11];
    float kernel2[11][11];
    float kernel3[11][11];
    // bias
    int bias = 1;

    // random values between 0 and 1 for each kernel
    for(int i=0;i<11;i++){
        for(int j=0;j<11;j++){
            kernel1[i][j]=rand()%2;
            kernel2[i][j]=rand()%2;
            kernel3[i][j]=rand()%2;
        }
    }

    // convolution between input image layers matrices and kernels
    float layer_conv1[224][224];
    convolution(input1,kernel1,layer_conv1,224,11);
    
}