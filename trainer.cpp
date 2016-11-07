#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <time.h>

double learningRate = 0.01;
double weightDecay = 0.005;
double batchSize = 128;

using namespace std;

/*
* upsamples the values to backpropagate to the previous layer
*
* @param backPropagatedError: is the matrix of errors coming from the next layer
* @param scale: is a matrix of 1s the size of the current layer
*/
vector<vector<double>> upsample(vector<vector<double>> backPropagatedError, vector<vector<double>> scale){
    cout << "Upsampling..." << endl;
    int height1 = backPropagatedError.size();
    int width1 = backPropagatedError[0].size();
    int height2 = scale.size();
    int width2 = scale[0].size();
    int row, column;
    vector<vector<double>> upsampleValuesMatrix(height1 * height2, vector<double>(width1 * width2));
    if (scale.size() % backPropagatedError.size() != 0 || scale[0].size() % backPropagatedError[0].size() != 0){
        vector<vector<double>> rescale(backPropagatedError.size() * 2, vector<double>(backPropagatedError[0].size() * 2));
        vector<vector<double>> temp(height1 * rescale.size(), vector<double>(width1 * rescale[0].size()));
        for (int i = 0; i < rescale.size(); i++){
            for (int j = 0; j < rescale[0].size(); j++){
                rescale[i][j] = 1;
            }
        }
        #pragma omp parallel num_threads(4)
        {
            #pragma omp for collapse(4)
            for (int i = 0; i < height1; i++){
                for (int j = 0; j < width1; j++){
                    for (int k = 0; k < rescale.size(); k++){
                        for (int z = 0; z < rescale[0].size(); z++){
                            row = i * height2 + k;
                            column = j * width2 + z;
                            temp[row][column] = backPropagatedError[i][j] * rescale[k][z];
                        }
                    }
                }
            }
            #pragma for collapse(2)
            for (int i = 0; i < upsampleValuesMatrix.size(); i++){
                for (int j = 0; j < upsampleValuesMatrix[0].size(); j++){
                    upsampleValuesMatrix[i][j] = temp[(temp.size() / 2) + i][(temp[0].size() / 2) + j];
                }
            }
        }
    }
    else{
        #pragma omp parallel num_threads(4)
        {
            #pragma omp for collapse(4)
            for (int i = 0; i < height1; i++){
                for (int j = 0; j < width1; j++){
                    for (int k = 0; k < height2; k++){
                        for (int z = 0; z < width2; z++){
                            row = i * height2 + k;
                            column = j * width2 + z;
                            upsampleValuesMatrix[row][column] = backPropagatedError[i][j] * scale[k][z];
                        }
                    }
                }
            }
        }
    }
    cout << "...upsampling done." << endl;
    return upsampleValuesMatrix;
}

/*
* Calculates the values to back propagated from the last layer (output layer)
* @param layerHeight: height of the layer
* @param layerWidth: width of the layer
* @param desiredOutput: desired output of the network
* @param output: actual output of the network
*   
*/
vector<vector<double>> lastLayerBackValues(int layerHeight, int layerWidth, double desiredOutput, double output){   
    cout << "Calculating back values for last layer..." << endl;
    vector<vector<double>> backValues(2, vector<double>(1));
    for (int i = 0; i < layerHeight; i++){
        for (int j = 0; j < layerWidth; j++){
            backValues[i][j] = -(desiredOutput - max((double)0, output)) * max((double)0, output) * (1 - max((double)0, output));
        }
    }
    cout << "...last layer back values calculation complete." << endl;
    return backValues;
}

/*
* Calculates the values to back propagate from a generic layerBackValues
* @param weights: weights matrix of the current layer
* @param nextLayerErrors: Delta errors coming from the next layer
* @param layerOutput: actual output of the current layer
*/
vector<vector<double>> layerBackValues(vector<vector<double>> weights, vector<vector<double>> nextLayerErrors, vector<vector<double>> layerOutput){
    cout << "Calculating back values..." << endl;
    vector<vector<double>> backValues(weights.size(), vector<double>(weights[0].size()));
    vector<vector<double>> errors;
    if (weights.size() != nextLayerErrors.size() || weights[0].size() != nextLayerErrors[0].size()){
        vector<vector<double>> scale(weights.size(), vector<double>(weights[0].size()));
        for (int i = 0; i < weights.size(); i++){
            for (int j = 0; j < weights[0].size(); j++){
                scale[i][j] = 1;
            }
        }
        errors = upsample(nextLayerErrors, scale);
    }
    else    errors = nextLayerErrors;
    for (int i = 0; i < weights.size(); i++){
        for (int j = 0; j < weights[0].size(); j++){
            backValues[i][j] = weights[i][j] * errors[i][j] * max((double)0, layerOutput[i][j]) * (1 - max((double)0, layerOutput[i][j]));
        }
    }
    cout << "...layer back values calculation complete." << endl;
    return backValues;
}

/*
* backPropagation algorithm implementation, calculates the matrix of values needed for gradient computation
* @param forwardValues: is the output of the layer
* @param backWardValues: are the values coming from the next layer
* @param params: are the parameters to modify (wieghts or biases)
*/
vector<vector<double>> backPropagation(vector<vector<double>> forwardValues, vector<vector<double>> backwardValues, vector<vector<double>> params){
    cout << "Calculating errors..." << endl;
    int count = 0;
    vector<vector<double>> deltaErrors(backwardValues.size(), vector<double>(backwardValues[0].size()));
    vector<vector<int>> identity(forwardValues.size(), vector<int>(forwardValues[0].size()));
    clock_t begin;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < forwardValues.size(); i++){
            for (int j = 0; j < forwardValues[0].size(); j++){
                if (i == j)     identity[i][j] = 1;
                else            identity[i][j] = 0;
            }
        }
        begin = clock();
        #pragma omp for collapse(2)
        for (int i = 0; i < backwardValues.size(); i++){
            for (int j = 0; j < backwardValues[0].size(); j++){
                count++;
                deltaErrors[i][j] = params[j][i] * backwardValues[i][j] * max((double)0, forwardValues[i][j]) * (identity[i][j] - max((double)0, forwardValues[i][j]));
                //cout << '\r' << "Error " << setw(5) << count << " calculated" << flush;
            }
        }
    }
    cout << endl << "Time elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
    cout << endl;
    cout << "...errors done." << endl;
    return deltaErrors;
}

/*
* batch gradient descent implementation, calculates the gradients for a layer and updates the weight and biases in the matrices
*
* @param forwardValues: is the output of the layer
* @param backWardValues: are the values coming from the next layer
* @param layerWeights: weights of the current layer
* @param layerBiases: biases of the current layer
*/
map<char, vector<vector<double>>> layerUpdater(vector<vector<double>> forwardValues, vector<vector<double>> backwardValues, vector<vector<double>> layerWeights, vector<vector<double>> layerBiases){
    cout << "Updating values..." << endl;
    vector<vector<double>> deltaW(backwardValues.size(), vector<double>(backwardValues[0].size()));
    vector<vector<double>> deltaB(backwardValues.size(), vector<double>(backwardValues[0].size()));
    vector<vector<double>> temp1;
    vector<vector<double>> temp2;
    map<char, vector<vector<double>>> paramMap;
    clock_t begin = clock();
    for (int b = 0; b < batchSize; b++){
        temp1 = backPropagation(forwardValues, backwardValues, layerWeights);
        temp2 = backPropagation(forwardValues, backwardValues, layerWeights);
        #pragma omp parallel num_threads(4)
        {
            #pragma omp for collapse(2)
            for (int i = 0; i < backwardValues.size(); i++){
                for (int j = 0; j < backwardValues[0].size(); j++){
                    deltaW[i][j] += temp1[i][j];
                    deltaB[i][j] += temp2[i][j];
                }
            }
        }
        #pragma omp for collapse(2)
        for (int i = 0; i < backwardValues.size(); i++){
            for (int j = 0; j < backwardValues[0].size(); j++){
                layerWeights[i][j] = layerWeights[i][j] - (learningRate * (((1 / batchSize) * deltaW[i][j]) + weightDecay * layerWeights[i][j]));
                layerBiases[i][j] = layerBiases[i][j] - (learningRate * ((1 / batchSize) * deltaB[i][j]));
            }
        }
    }
    paramMap['W'] = layerWeights;
    paramMap['B'] = layerBiases;
    cout << "...update finished." << endl;
    cout << "Total time elapsed: " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
    return paramMap;
}

//Test

/*
int main()
{
    default_random_engine gen;
    normal_distribution<double> distr(0, 100);

    vector<vector<double>> forwardValues(55, vector<double>(55));
    vector<vector<double>> backwardValues(55, vector<double>(55));
    vector<vector<double>> layerWeights(55, vector<double>(55));
    vector<vector<double>> layerBiases(55, vector<double>(55));

    for (int i = 0; i < 55; i++)
    {
        for (int j = 0; j < 55; j++)
        {
            forwardValues[i][j] = distr(gen);
            backwardValues[i][j] = distr(gen);
            layerWeights[i][j] = distr(gen);
            layerBiases[i][j] = distr(gen);
        }
    }
    lastLayerBackValues(2, 1, 0.5048, 0.99);
    layerUpdater(forwardValues,layerBackValues(forwardValues, backwardValues, layerWeights),layerWeights,layerBiases);

    /*
    vector<vector<vector<double>>> forward{{{1, 2, 3}, {1.3, 1.6, 6.4}, {3.1, 4, 2}},
                                           {{1.3, 1.6, 6.4}, {2.3, 3, 1}, {1, 8.34, 3.456}},
                                           {{1, 6, 2.5}, {3.76, 3, 9.765}, {3, 8, 2.234}}};
    

    default_random_engine gen;
    normal_distribution<double> distr(0, 100);

    vector<vector<double>> forwardValues(55, vector<double>(55));            
    vector<vector<double>> backwardValues(55, vector<double>(55));
    vector<vector<double>> layerWeights(55, vector<double>(55));
    vector<vector<double>> layerBiases(55, vector<double>(55));

    for(int i=0;i<55;i++){
        for(int j=0;j<55;j++){
            forwardValues[i][j]=distr(gen);
            backwardValues[i][j]=distr(gen);
            layerWeights[i][j]=distr(gen);
            layerBiases[i][j]=distr(gen);
        }
    }
    
    int count=0;
    cout << "weights: " << endl;
    for(int i=0;i<layerWeights.size();i++){
        for(int j=0;j<layerWeights[0].size();j++){
            count++;
            cout << setw(10) <<layerWeights[i][j] << " ";
            if(count==layerWeights[0].size()){
                cout << endl;
                count=0;
            } 
        }
    }

    cout << "biases: " << endl;
    for(int i=0;i<layerBiases.size();i++){
        for(int j=0;j<layerBiases[0].size();j++){
            count++;
            cout << setw(10) <<layerBiases[i][j] << " ";
            if(count==layerBiases[0].size()){
                cout << endl;
                count=0;
            } 
        }
    }

    map <char, vector<vector<double>>> mp=layerUpdater(forwardValues, backwardValues, layerWeights, layerBiases);
    vector<vector<double>> w=mp['W'];
    vector<vector<double>> b=mp['B'];
    count=0;
    cout << "weights: " << endl;
    for(int i=0;i<w.size();i++){
        for(int j=0;j<w[0].size();j++){
            count++;
            cout << setw(10) << w[i][j] << " ";
            if(count==w[0].size()){
                cout << endl;
                count=0;
            } 
        }
    }
    count=0;
    cout << "biases: " << endl;
    for(int i=0;i<b.size();i++){
        for(int j=0;j<b[0].size();j++){
            count++;
            cout << setw(10) <<b[i][j] << " ";
            if(count==b[0].size()){
                cout << endl;
                count=0;
            } 
        }
    }

        vector<vector<int>> m1{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        vector<vector<int>> m2{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        int i, j;
        int count = 0;
        vector<vector<int>> up = upsample(m1, m2);
        for (i = 0; i < up.size(); i++)
        {
            for (j = 0; j < up.size(); j++)
            {
                count += 1;
                cout << up[i][j] << " ";
                if (count == up.size())
                {
                    count = 0;
                    cout << endl;
                }
            }
        }
    }
}*/
