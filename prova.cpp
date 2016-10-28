#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace std;



/* convolution function
   input: input matrix, kernel matrix, output matrix, the stride and the bias */
void convolution(vector<vector<vector<float>>> input, vector<vector<vector<vector<float>>>> kernel, vector<vector<vector<float>>> &output, int stride, int bias){
    

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
                                output[f][oy][ox] += input[i][iy][ix] * kernel[f][i][ky][kx];
                            }
                            ix++;
                        }
                        ix=ox*stride;
                        iy++;
                    }
                    iy=oy*stride;
                    output[f][oy][ox] += bias;
                }
            }
        }
    }

}



/* ReLU non-linear function: apply the ReLU nonlinear function f(x)=max(0,x) to each input value
   input: matrix on which applying ReLU function */
void relu(vector<vector<vector<float>>> &relu){

    int feat=relu.size();
    int size=relu[0].size();

    for(int f=0;f<feat;f++){

        for(int i=0;i<size;i++){

            for(int j=0;j<size;j++){
                
                relu[f][i][j]=max((float)0, relu[f][i][j]);
            }
        }
    }

}



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, dimension of the overlapping pooling matrix and the stride */
void maxpooling(vector<vector<vector<float>>> input, vector<vector<vector<float>>> &output){
    
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

}


void f(int &i){
    i++;
}


int main() {

    vector<int> r(2);
    int j=0;
    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                f(j);
                r[omp_get_thread_num()]=j;
                cout<<omp_get_thread_num()<<endl;
            }

            #pragma omp section
            {
                j=0;
                f(j);
                f(j);
                r[omp_get_thread_num()]=j;
                cout<<omp_get_thread_num()<<endl;
            }
        }
    }
    cout<<r[0]<<endl;
    cout<<r[1]<<endl;

    // Gaussian distribution with zero-mean and standard deviation 0.01
    std::default_random_engine gen;
    std::normal_distribution<float> distr(0,0.01);

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
    const int feat1=48;

    // 3 kernels which will convolved on the 3 image matrices (11x11x3xfeat1)
    vector<vector<vector<vector<float>>>> kernel1(feat1, vector<vector<vector<float>>>(3, vector<vector<float>>(11, vector<float>(11))));
    

    // initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
    for(int f=0;f<feat1;f++){
        for(int i=0;i<3;i++){
            for(int j=0;j<11;j++){
                for(int k=0;k<11;k++){
                    
                    kernel1[f][i][j][k]=distr(gen);
                }
            }
        }
    }


    // output of convolutional operation between input and kernel of layer 1 (55x55xfeat1)
    vector<vector<vector<float>>> layer1(feat1, vector<vector<float>>(55, vector<float>(55)));


    int stride1 = round((float)(input[0].size() - kernel1[0][0].size())/(layer1[0].size()-1));
    
    
    // convolution between input image layers and kernels
    //convolution(input,kernel1,layer1,stride1,0);
    
    // ReLU nonlinearity
    //relu(layer1);
    


    // LAYER 2

    // number of features
    const int feat2=128;

    // kernel (5x5xfeat1xfeat2)
    vector<vector<vector<vector<float>>>> kernel2(feat2, vector<vector<vector<float>>>(feat1, vector<vector<float>>(5, vector<float>(5))));
    
    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat2;f++){
        for(int i=0;i<feat1;i++){
            for(int j=0;j<5;j++){
                for(int k=0;k<5;k++){
                    
                    kernel2[f][i][j][k]=distr(gen);
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
    //convolution(layer1,kernel2,conv2,stride2,1);
    
    // ReLU nonlinearity
    //relu(conv2);
    
    // overlapped max-pooling    
    //maxpooling(conv2,layer2);
    


    // LAYER 3

    // number of features
    const int feat3=192;

    // kernel (3x3xfeat2xfeat3)
    vector<vector<vector<vector<float>>>> kernel3(feat3, vector<vector<vector<float>>>(feat2, vector<vector<float>>(3, vector<float>(3))));
    
    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat3;f++){
        for(int i=0;i<feat2;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    
                    kernel3[f][i][j][k]=distr(gen);
                }
            }
        }
    }

    // output of convolutional operation between layer2 and kernel of layer 3 (27x27xfeat3)
    vector<vector<vector<float>>> conv3(feat3, vector<vector<float>>(27, vector<float>(27)));

    // output of overlapped max-pooling operation (13x13xfeat3)
    vector<vector<vector<float>>> layer3(feat3, vector<vector<float>>(13, vector<float>(13)));


    int stride3 = round((float)(layer2[0].size() - kernel3[0][0].size())/(conv3[0].size()-1));
    
    // convolution between layer 2 and kernels of layer 3
    //convolution(layer2,kernel3,conv3,stride3,0);
    
    // ReLU nonlinearity
    //relu(conv3);
    
    // overlapped max-pooling
    //maxpooling(conv3,layer3);
    


    // LAYER 4

    // number of features
    const int feat4=192;

    // kernel (3x3xfeat3xfeat4)
    vector<vector<vector<vector<float>>>> kernel4(feat4, vector<vector<vector<float>>>(feat3, vector<vector<float>>(3, vector<float>(3))));

    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat4;f++){
        for(int i=0;i<feat3;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    
                    kernel4[f][i][j][k]=distr(gen);
                }
            }
        }
    }

    // output of convolutional operation between layer3 and kernel of layer 4 (13x13xfeat4)
    vector<vector<vector<float>>> layer4(feat4, vector<vector<float>>(13, vector<float>(13)));


    int stride4 = round((float)(layer3[0].size() - kernel4[0][0].size())/(layer4[0].size()-1));
    
    // convolution between layer 3 and kernels of layer 4
    //convolution(layer3,kernel4,layer4,stride4,1);
    
    // ReLU nonlinearity
    //relu(layer4);
    


    // LAYER 5

    // number of features
    const int feat5=128;

    // kernel (3x3xfeat3xfeat4)
    vector<vector<vector<vector<float>>>> kernel5(feat5, vector<vector<vector<float>>>(feat4, vector<vector<float>>(3, vector<float>(3))));

    // random values between -2 and 2 for each kernel
    for(int f=0;f<feat5;f++){
        for(int i=0;i<feat4;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    
                    kernel5[f][i][j][k]=distr(gen);
                }
            }
        }
    }

    // output of convolutional operation between layer4 and kernel of layer 5 (13x13xfeat5)
    vector<vector<vector<float>>> layer5(feat5, vector<vector<float>>(13, vector<float>(13)));

    int stride5 = round((float)(layer4[0].size() - kernel5[0][0].size())/(layer5[0].size()-1));
    
    // convolution between layer 4 and kernels of layer 5
    //convolution(layer4,kernel5,layer5,stride5,1);
    
    // ReLU nonlinearity
    //relu(layer5);
    


    
    // LAYER 6 - FULLY-CONNECTED

    // layer 6 fully-connected
    vector<float> layer6(2048);

    // overlapped max-pooling (layer5 -> 6x6xfeat5)
    vector<vector<vector<float>>> pool6(feat5, vector<vector<float>>(6, vector<float>(6)));
    
    //maxpooling(layer5,pool6);
    
    // weights matrix (2048 weights x 6x6xfeat5 input neurons)
    vector<vector<vector<vector<float>>>> weight6(2048, vector<vector<vector<float>>>(feat5, vector<vector<float>>(6, vector<float>(6))));

    // random values between -2 and 2 for each kernel
    for(int f=0;f<2048;f++){
        for(int i=0;i<feat5;i++){
            for(int j=0;j<6;j++){
                for(int k=0;k<6;k++){
                    
                    weight6[f][i][j][k]=distr(gen);
                }
            }
        }
    }

    // from layer5 to fully-connected layer6
    for(int i=0;i<2048;i++){

        for(int j=0; j<feat5;j++){

            for(int k=0;k<6;k++){

                for(int l=0;l<6;l++){

                    //layer6[i] = weight6[i][j][k][l]*pool6[j][k][l];
                }
            }
        }
        //layer6[i]+=1;
        //layer6[i] = max((float)0, layer6[i]);   // ReLU function
    }
    
    


    // LAYER 7

    // layer 7 fully-connected
    vector<float> layer7(2048);

    // weights matrix (2048 weights x 6x6xfeat5 input neurons)
    vector<float> weight7(2048);

    // random values between -2 and 2 for each kernel
    for(int i=0;i<2048;i++){
                    
        weight7[i]=distr(gen);
    }

    // from layer6 to layer 7
    for(int i=0;i<2048;i++){

        for(int j=0;j<2048;j++){

            //layer7[i] = weight7[j]*layer6[j];
        }
        //layer7[i]+=1;
        //layer7[i] = max((float)0, layer7[i]);   // ReLU function
    }

    cout<<"end"<<endl;

}