#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include </home/lorenzo/OpenMP-CNN/input_reader.cpp>
#include </home/lorenzo/OpenMP-CNN/trainer.cpp>
#include <algorithm>
#include <map>

#include <time.h>

using namespace std;


/* convolution function
   input: input matrix, kernel matrix, output matrix, the stride and the bias */
void convolution(vector<vector<vector<double>>> input, vector<vector<vector<vector<double>>>> kernel, vector<vector<vector<double>>> &output, int stride, int bias){
    
    #pragma omp parallel num_threads(4)
    {
        int iy,ix;  // input y and x coordinates
        int feat=output.size();
        int depth=input.size();
        int isize=input[0].size();
        int ksize=kernel[0][0].size();
        int osize=output[0].size();

        #pragma omp for collapse(3)
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
    
}



/* ReLU non-linear function: apply the ReLU nonlinear function f(x)=max(0,x) to each input value
   input: matrix on which applying ReLU function */
void relu(vector<vector<vector<double>>> &relu){

    #pragma omp parallel num_threads(4)
    {
        int feat=relu.size();
        int size=relu[0].size();

        #pragma omp for collapse(3)
        for(int f=0;f<feat;f++){

            for(int i=0;i<size;i++){

                for(int j=0;j<size;j++){
                    
                    relu[f][i][j]=max((double)0, relu[f][i][j]);
                }
            }
        }
    }

}



/* overlapping max-pooling funtion: extract the maximum value from the output of the convolutional operation
   input: input matrix, output matrix, dimension of the overlapping pooling matrix and the stride */
void maxpooling(vector<vector<vector<double>>> input, vector<vector<vector<double>>> &output){
    
    #pragma omp parallel num_threads(4)
    {
        int p=3;    //max-pooling matrix size (3x3)
        int stride=2;   //overlapping: stride < p
        int feat=input.size();
        int isize=input[0].size();
        int osize=output[0].size();

        #pragma omp for collapse(3)
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

}



int main() {

    // Gaussian distribution with zero-mean and standard deviation 0.01
    std::default_random_engine gen;
    std::normal_distribution<double> distr(0,0.01);


    // input image sample (224x224x3)
    vector<vector<vector<double>>> input(3, vector<vector<double>>(224, vector<double>(224)));
    /*
    // random bit values for each image layer
    for(int i=0;i<3;i++){
        for(int j=0;j<224;j++){
            for(int k=0;k<224;k++){

                input[i][j][k]=rand()%255;
            }
        }
    }*/
    
   map<char, vector<vector<double>>> m=getColors("n01503061_3.txt");
    input[0]=m['R'];
    input[1]=m['G'];
    input[2]=m['B'];

    const clock_t begin_time = clock();


    const int feat1=48;     //number of features for layer 1
    const int feat2=128;    //number of features for layer 2
    const int feat3=192;    //number of features for layer 3
    const int feat4=192;    //number of features for layer 4
    const int feat5=128;    //number of features for layer 5

    //shared array for saving output of layers 2
    vector<vector<vector<vector<double>>>> out2(2,vector<vector<vector<double>>>(feat2, vector<vector<double>>(27, vector<double>(27))));
    //shared array for saving output of layers 5
    vector<vector<vector<vector<double>>>> out5(2,vector<vector<vector<double>>>(feat5, vector<vector<double>>(13, vector<double>(13))));
    //shared array for saving output of layers 6 and 7
    vector<vector<double>> out67(2,vector<double>(2048));

    omp_set_nested(1);
    #pragma omp parallel num_threads(2)
    {
        //layers 1 and 2
        #pragma omp sections
        {
            //first CNN
            #pragma omp section
            {
                // LAYER 1

                //3 kernels which will convolved on the 3 image matrices (11x11x3xfeat1)
                vector<vector<vector<vector<double>>>> kernel1(feat1, vector<vector<vector<double>>>(3, vector<vector<double>>(11, vector<double>(11))));
                
                //output of convolutional operation between input and kernel of layer 1 (55x55xfeat1)
                vector<vector<vector<double>>> layer1(feat1, vector<vector<double>>(55, vector<double>(55)));

                int stride1 = round((double)(input[0].size() - kernel1[0][0].size())/(layer1[0].size()-1));

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat1;f++){
                        for(int i=0;i<3;i++){
                            for(int j=0;j<11;j++){
                                for(int k=0;k<11;k++){
                                    
                                    kernel1[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }
                //convolution between input image layers and kernels
                convolution(input,kernel1,layer1,stride1,0);
                
                //ReLU nonlinearity
                relu(layer1);



                // LAYER 2
                
                //kernel (5x5xfeat1xfeat2)
                vector<vector<vector<vector<double>>>> kernel2(feat2, vector<vector<vector<double>>>(feat1, vector<vector<double>>(5, vector<double>(5))));

                //output of convolutional operation between layer1 and kernel of layer 2 (55x55xfeat2)
                vector<vector<vector<double>>> conv2(feat2, vector<vector<double>>(55, vector<double>(55)));

                //output of overlapped max-pooling operation (27x27xfeat2)
                vector<vector<vector<double>>> layer2(feat2, vector<vector<double>>(27, vector<double>(27)));

                int stride2 = round((double)(layer1[0].size() - kernel2[0][0].size())/(conv2[0].size()-1));
                
                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat2;f++){
                        for(int i=0;i<feat1;i++){
                            for(int j=0;j<5;j++){
                                for(int k=0;k<5;k++){
                                    
                                    kernel2[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                //convolution between layer 1 and kernels of layer 2
                convolution(layer1,kernel2,conv2,stride2,1);
                
                //ReLU nonlinearity
                relu(conv2);
                
                //overlapped max-pooling    
                maxpooling(conv2,layer2);
                

                //save output of layer 2 for the first CNN
                out2[0]=layer2;
            }

            //second CNN
            #pragma omp section
            {
                // LAYER 1
                
                //3 kernels which will convolved on the 3 image matrices (11x11x3xfeat1)
                vector<vector<vector<vector<double>>>> kernel1(feat1, vector<vector<vector<double>>>(3, vector<vector<double>>(11, vector<double>(11))));
                
                //output of convolutional operation between input and kernel of layer 1 (55x55xfeat1)
                vector<vector<vector<double>>> layer1(feat1, vector<vector<double>>(55, vector<double>(55)));

                int stride1 = round((double)(input[0].size() - kernel1[0][0].size())/(layer1[0].size()-1));

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat1;f++){
                        for(int i=0;i<3;i++){
                            for(int j=0;j<11;j++){
                                for(int k=0;k<11;k++){
                                    
                                    kernel1[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }
                //convolution between input image layers and kernels
                convolution(input,kernel1,layer1,stride1,0);
                
                //ReLU nonlinearity
                relu(layer1);



                // LAYER 2
                
                //kernel (5x5xfeat1xfeat2)
                vector<vector<vector<vector<double>>>> kernel2(feat2, vector<vector<vector<double>>>(feat1, vector<vector<double>>(5, vector<double>(5))));

                //output of convolutional operation between layer1 and kernel of layer 2 (55x55xfeat2)
                vector<vector<vector<double>>> conv2(feat2, vector<vector<double>>(55, vector<double>(55)));

                //output of overlapped max-pooling operation (27x27xfeat2)
                vector<vector<vector<double>>> layer2(feat2, vector<vector<double>>(27, vector<double>(27)));

                int stride2 = round((double)(layer1[0].size() - kernel2[0][0].size())/(conv2[0].size()-1));
                
                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat2;f++){
                        for(int i=0;i<feat1;i++){
                            for(int j=0;j<5;j++){
                                for(int k=0;k<5;k++){
                                    
                                    kernel2[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                //convolution between layer 1 and kernels of layer 2
                convolution(layer1,kernel2,conv2,stride2,1);
                
                //ReLU nonlinearity
                relu(conv2);
                
                //overlapped max-pooling    
                maxpooling(conv2,layer2);
                

                //save output of layer 2 for the second CNN
                out2[1]=layer2;
            }

        }


        //layers 3, 4 and 5
        #pragma omp sections
        {
            //first CNN
            #pragma omp section
            {
                // LAYER 3

                //take output of layers 2 from both CNNs
                vector<vector<vector<double>>> layer21=out2[0];
                vector<vector<vector<double>>> layer22=out2[1];
                

                //kernels (3x3xfeat2xfeat3)
                vector<vector<vector<vector<double>>>> kernel31(feat3, vector<vector<vector<double>>>(feat2, vector<vector<double>>(3, vector<double>(3))));
                vector<vector<vector<vector<double>>>> kernel32(feat3, vector<vector<vector<double>>>(feat2, vector<vector<double>>(3, vector<double>(3))));

                //output of convolutional operation between layer2 and kernel of layer 3 (27x27xfeat3)
                vector<vector<vector<double>>> conv3(feat3, vector<vector<double>>(27, vector<double>(27)));

                //output of overlapped max-pooling operation (13x13xfeat3)
                vector<vector<vector<double>>> layer3(feat3, vector<vector<double>>(13, vector<double>(13)));

                int stride3 = round((double)(layer21[0].size() - kernel31[0][0].size())/(conv3[0].size()-1));
                
                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat3;f++){
                        for(int i=0;i<feat2;i++){
                            for(int j=0;j<3;j++){
                                for(int k=0;k<3;k++){
                                    
                                    kernel31[f][i][j][k]=distr(gen);
                                    kernel32[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }


                //convolution between layers 2 and kernels of layer 3
                convolution(layer21,kernel31,conv3,stride3,0);
                convolution(layer22,kernel32,conv3,stride3,0);
                
                // ReLU nonlinearity
                relu(conv3);
                
                //overlapped max-pooling
                maxpooling(conv3,layer3);



                // LAYER 4

                //kernel (3x3xfeat3xfeat4)
                vector<vector<vector<vector<double>>>> kernel4(feat4, vector<vector<vector<double>>>(feat3, vector<vector<double>>(3, vector<double>(3))));

                //output of convolutional operation between layer3 and kernel of layer 4 (13x13xfeat4)
                vector<vector<vector<double>>> layer4(feat4, vector<vector<double>>(13, vector<double>(13)));

                int stride4 = round((double)(layer3[0].size() - kernel4[0][0].size())/(layer4[0].size()-1));

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat4;f++){
                        for(int i=0;i<feat3;i++){
                            for(int j=0;j<3;j++){
                                for(int k=0;k<3;k++){
                                    
                                    kernel4[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                //convolution between layer 3 and kernels of layer 4
                convolution(layer3,kernel4,layer4,stride4,1);
                
                //ReLU nonlinearity
                relu(layer4);                
                


                // LAYER 5

                // kernel (3x3xfeat3xfeat4)
                vector<vector<vector<vector<double>>>> kernel5(feat5, vector<vector<vector<double>>>(feat4, vector<vector<double>>(3, vector<double>(3))));

                // output of convolutional operation between layer4 and kernel of layer 5 (13x13xfeat5)
                vector<vector<vector<double>>> layer5(feat5, vector<vector<double>>(13, vector<double>(13)));

                int stride5 = round((double)(layer4[0].size() - kernel5[0][0].size())/(layer5[0].size()-1));

                // initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat5;f++){
                        for(int i=0;i<feat4;i++){
                            for(int j=0;j<3;j++){
                                for(int k=0;k<3;k++){
                                    
                                    kernel5[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                
                // convolution between layer 4 and kernels of layer 5
                convolution(layer4,kernel5,layer5,stride5,1);
                
                // ReLU nonlinearity
                relu(layer5);
                

                //save output of layer 5 for the first CNN
                out5[0]=layer5;
            }


            //second CNN
            #pragma omp section
            {
                // LAYER 3

                //take output of layers 2 from both CNNs
                vector<vector<vector<double>>> layer21=out2[0];
                vector<vector<vector<double>>> layer22=out2[1];
                

                //kernels (3x3xfeat2xfeat3)
                vector<vector<vector<vector<double>>>> kernel31(feat3, vector<vector<vector<double>>>(feat2, vector<vector<double>>(3, vector<double>(3))));
                vector<vector<vector<vector<double>>>> kernel32(feat3, vector<vector<vector<double>>>(feat2, vector<vector<double>>(3, vector<double>(3))));

                //output of convolutional operation between layer2 and kernel of layer 3 (27x27xfeat3)
                vector<vector<vector<double>>> conv3(feat3, vector<vector<double>>(27, vector<double>(27)));

                //output of overlapped max-pooling operation (13x13xfeat3)
                vector<vector<vector<double>>> layer3(feat3, vector<vector<double>>(13, vector<double>(13)));

                int stride3 = round((double)(layer21[0].size() - kernel31[0][0].size())/(conv3[0].size()-1));
                
                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat3;f++){
                        for(int i=0;i<feat2;i++){
                            for(int j=0;j<3;j++){
                                for(int k=0;k<3;k++){
                                    
                                    kernel31[f][i][j][k]=distr(gen);
                                    kernel32[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }


                //convolution between layers 2 and kernels of layer 3
                convolution(layer21,kernel31,conv3,stride3,0);
                convolution(layer22,kernel32,conv3,stride3,0);
                
                // ReLU nonlinearity
                relu(conv3);
                
                //overlapped max-pooling
                maxpooling(conv3,layer3);



                // LAYER 4

                //kernel (3x3xfeat3xfeat4)
                vector<vector<vector<vector<double>>>> kernel4(feat4, vector<vector<vector<double>>>(feat3, vector<vector<double>>(3, vector<double>(3))));

                //output of convolutional operation between layer3 and kernel of layer 4 (13x13xfeat4)
                vector<vector<vector<double>>> layer4(feat4, vector<vector<double>>(13, vector<double>(13)));

                int stride4 = round((double)(layer3[0].size() - kernel4[0][0].size())/(layer4[0].size()-1));

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat4;f++){
                        for(int i=0;i<feat3;i++){
                            for(int j=0;j<3;j++){
                                for(int k=0;k<3;k++){
                                    
                                    kernel4[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                //convolution between layer 3 and kernels of layer 4
                convolution(layer3,kernel4,layer4,stride4,1);
                
                //ReLU nonlinearity
                relu(layer4);                
                


                // LAYER 5

                // kernel (3x3xfeat3xfeat4)
                vector<vector<vector<vector<double>>>> kernel5(feat5, vector<vector<vector<double>>>(feat4, vector<vector<double>>(3, vector<double>(3))));

                // output of convolutional operation between layer4 and kernel of layer 5 (13x13xfeat5)
                vector<vector<vector<double>>> layer5(feat5, vector<vector<double>>(13, vector<double>(13)));

                int stride5 = round((double)(layer4[0].size() - kernel5[0][0].size())/(layer5[0].size()-1));

                // initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<feat5;f++){
                        for(int i=0;i<feat4;i++){
                            for(int j=0;j<3;j++){
                                for(int k=0;k<3;k++){
                                    
                                    kernel5[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                
                // convolution between layer 4 and kernels of layer 5
                convolution(layer4,kernel5,layer5,stride5,1);
                
                // ReLU nonlinearity
                relu(layer5);
                    

                //save output of layer 5 for the second CNN
                out5[1]=layer5;
            }
        }

        

        //layers 6
        #pragma omp sections
        {
            //first CNN
            #pragma omp section
            {
                // LAYER 6

                //take output of layers 5 from both CNNs
                vector<vector<vector<double>>> layer51=out5[0];
                vector<vector<vector<double>>> layer52=out5[1];                


                //weights matrix (2048 weights x 6x6xfeat5 input neurons)
                vector<vector<vector<vector<double>>>> weight61(2048, vector<vector<vector<double>>>(feat5, vector<vector<double>>(6, vector<double>(6))));
                vector<vector<vector<vector<double>>>> weight62(2048, vector<vector<vector<double>>>(feat5, vector<vector<double>>(6, vector<double>(6))));

                //overlapped max-pooling (layer5 -> 6x6xfeat5)
                vector<vector<vector<double>>> pool61(feat5, vector<vector<double>>(6, vector<double>(6)));
                vector<vector<vector<double>>> pool62(feat5, vector<vector<double>>(6, vector<double>(6)));

                //layer 6 fully-connected
                vector<double> layer6(2048);

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<2048;f++){
                        for(int i=0;i<feat5;i++){
                            for(int j=0;j<6;j++){
                                for(int k=0;k<6;k++){
                                    
                                    weight61[f][i][j][k]=distr(gen);
                                    weight62[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                maxpooling(layer51,pool61);
                maxpooling(layer52,pool62);
                

                //from layer5 to fully-connected layer6
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for
                    for(int i=0;i<2048;i++){

                        for(int j=0; j<feat5;j++){

                            for(int k=0;k<6;k++){

                                for(int l=0;l<6;l++){

                                    layer6[i] = weight61[i][j][k][l]*pool61[j][k][l];
                                }
                            }
                        }
                        layer6[i]+=1;
                        layer6[i] = max((double)0, layer6[i]);   //ReLU function
                    }
                    
                    #pragma omp barrier

                    #pragma omp for
                    for(int i=0;i<2048;i++){
                       
                        for(int j=0; j<feat5;j++){

                            for(int k=0;k<6;k++){

                                for(int l=0;l<6;l++){

                                    layer6[i] = weight62[i][j][k][l]*pool62[j][k][l];
                                }
                            }
                        }
                        layer6[i]+=1;
                        layer6[i] = max((double)0, layer6[i]);   //ReLU function
                    }
                }
                
                //save output of layer 6 for the first CNN
                out67[0]=layer6;
            }

            //first CNN
            #pragma omp section
            {
                // LAYER 6

                //take output of layers 5 from both CNNs
                vector<vector<vector<double>>> layer51=out5[0];
                vector<vector<vector<double>>> layer52=out5[1];                


                //weights matrix (2048 weights x 6x6xfeat5 input neurons)
                vector<vector<vector<vector<double>>>> weight61(2048, vector<vector<vector<double>>>(feat5, vector<vector<double>>(6, vector<double>(6))));
                vector<vector<vector<vector<double>>>> weight62(2048, vector<vector<vector<double>>>(feat5, vector<vector<double>>(6, vector<double>(6))));

                //overlapped max-pooling (layer5 -> 6x6xfeat5)
                vector<vector<vector<double>>> pool61(feat5, vector<vector<double>>(6, vector<double>(6)));
                vector<vector<vector<double>>> pool62(feat5, vector<vector<double>>(6, vector<double>(6)));

                //layer 6 fully-connected
                vector<double> layer6(2048);

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(4)
                    for(int f=0;f<2048;f++){
                        for(int i=0;i<feat5;i++){
                            for(int j=0;j<6;j++){
                                for(int k=0;k<6;k++){
                                    
                                    weight61[f][i][j][k]=distr(gen);
                                    weight62[f][i][j][k]=distr(gen);
                                }
                            }
                        }
                    }
                }

                maxpooling(layer51,pool61);
                maxpooling(layer52,pool62);
                

                //from layer5 to fully-connected layer6
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for
                    for(int i=0;i<2048;i++){

                        for(int j=0; j<feat5;j++){

                            for(int k=0;k<6;k++){

                                for(int l=0;l<6;l++){

                                    layer6[i] = weight61[i][j][k][l]*pool61[j][k][l];
                                }
                            }
                        }
                        layer6[i]+=1;
                        layer6[i] = max((double)0, layer6[i]);   //ReLU function
                    }
                    
                    #pragma omp barrier

                    #pragma omp for
                    for(int i=0;i<2048;i++){
                       
                        for(int j=0; j<feat5;j++){

                            for(int k=0;k<6;k++){

                                for(int l=0;l<6;l++){

                                    layer6[i] = weight62[i][j][k][l]*pool62[j][k][l];
                                }
                            }
                        }
                        layer6[i]+=1;
                        layer6[i] = max((double)0, layer6[i]);   //ReLU function
                    }
                }
                
                //save output of layer 6 for the second CNN
                out67[1]=layer6;
            }

        }


        //layers 7
        #pragma omp sections
        {
            //first CNN
            #pragma omp section
            {
                // LAYER 7

                //take output of layers 6 from both CNNs
                vector<double> layer61=out67[0];
                vector<double> layer62=out67[1];


                //layer 7 fully-connected
                vector<double> layer7(2048);

                //weights matrix (2048 weights x 2048 input neurons)
                vector<vector <double>> weight71(2048, vector<double>(2048));
                vector<vector <double>> weight72(2048, vector<double>(2048));

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(2)
                    for(int i=0;i<2048;i++){
                        for(int j=0;j<2048;j++){

                            weight71[i][j]=distr(gen);
                            weight72[i][j]=distr(gen);
                        }
                    }
                }
                
                //from layer6 to layer 7
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for
                    for(int i=0;i<2048;i++){
                        for(int j=0;j<2048;j++){

                            layer7[i] = weight71[i][j]*layer61[j];
                        }
                        layer7[i]+=1;
                        layer7[i] = max((double)0, layer7[i]);   //ReLU function
                    }

                    #pragma omp barrier

                    #pragma omp for
                    for(int i=0;i<2048;i++){
                        for(int j=0;j<2048;j++){

                            layer7[i] = weight72[i][j]*layer62[j];
                        }
                        layer7[i]+=1;
                        layer7[i] = max((double)0, layer7[i]);   //ReLU function
                    }
                }
                

                //save output of layer 7 for the first CNN
                out67[0]=layer7;
            }


            //second CNN
            #pragma omp section
            {
                // LAYER 7

                //take output of layers 6 from both CNNs
                vector<double> layer61=out67[0];
                vector<double> layer62=out67[1];


                //layer 7 fully-connected
                vector<double> layer7(2048);

                //weights matrix (2048 weights x 2048 input neurons)
                vector<vector <double>> weight71(2048, vector<double>(2048));
                vector<vector <double>> weight72(2048, vector<double>(2048));

                //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for collapse(2)
                    for(int i=0;i<2048;i++){
                        for(int j=0;j<2048;j++){

                            weight71[i][j]=distr(gen);
                            weight72[i][j]=distr(gen);
                        }
                    }
                }
                
                //from layer6 to layer 7
                #pragma omp parallel num_threads(4)
                {
                    #pragma omp for
                    for(int i=0;i<2048;i++){
                        for(int j=0;j<2048;j++){

                            layer7[i] = weight71[i][j]*layer61[j];
                        }
                        layer7[i]+=1;
                        layer7[i] = max((double)0, layer7[i]);   //ReLU function
                    }

                    #pragma omp barrier

                    #pragma omp for
                    for(int i=0;i<2048;i++){
                        for(int j=0;j<2048;j++){

                            layer7[i] = weight72[i][j]*layer62[j];
                        }
                        layer7[i]+=1;
                        layer7[i] = max((double)0, layer7[i]);   //ReLU function
                    }
                }
                

                //save output of layer 7 for the second CNN
                out67[1]=layer7;
            }

        }
    }


    // LAYER 8

    //number of class labels
    const int lab=2;

    //take output of layers 6 from both CNNs
    vector<double> layer71=out67[0];
    vector<double> layer72=out67[1];

    //layer 8 fully-connected
    vector<double> layer8(lab);

    //weights matrix (1000 weights x 2048 input neurons)
    vector<vector<double>> weight81(lab, vector<double>(2048));
    vector<vector<double>> weight82(lab, vector<double>(2048));
    
    //initialize weights from a zero-mean Gaussian distribution with standard deviation 0.01
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for collapse(2)
        for(int i=0;i<lab;i++){

            for(int j=0;j<2048;j++){

                weight81[i][j]=distr(gen);
                weight82[i][j]=distr(gen);
            }
        }
    }

    //from layers 7 to layer 8
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for(int i=0;i<lab;i++){
            for(int j=0;j<2048;j++){

                layer8[i] = weight81[i][j]*layer71[j];
            }
            layer8[i]+=1;
            layer8[i] = max((double)0, layer8[i]);   //ReLU function
        }

        #pragma omp barrier
        
        #pragma omp for
        for(int i=0;i<lab;i++){
            for(int j=0;j<2048;j++){

                layer8[i] = weight82[i][j]*layer72[j];
            }
            layer8[i]+=1;
            layer8[i] = max((double)0, layer8[i]);   //ReLU function
        }
    }

    //softmax
    double sum=0;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for(int i=0;i<lab;i++){

            layer8[i] = exp(layer8[i]);
            sum += layer8[i];
        }

        #pragma omp barrier

        #pragma omp for
        for(int i=0;i<lab;i++){

            layer8[i] = layer8[i]/sum;
        }
    }
    
    lastLayerBackValues(layer8.size(), 1, layer8[0], 0.99);

    cout << layer8[0] << ", " << layer8[1] << endl;
    cout << double(clock() - begin_time)/CLOCKS_PER_SEC<< endl;
}