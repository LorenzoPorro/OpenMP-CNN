#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <time.h>


double learningRate=0.01;
double weightDecay=0.005;
double batchSize=128;

using namespace std;

/*
      * upsamples the values to backpropagate to the previous layer, needed only for the pooling layers
      *
      */
vector<vector<int>> upsample(vector<vector<int>> &backPropagatedError, vector<vector<int>> &derivativeOfAggregation)
{
    int height1 = backPropagatedError.size();
    int width1 = backPropagatedError[0].size();
    int height2 = derivativeOfAggregation.size();
    int width2 = derivativeOfAggregation[0].size();
    int row, column;
    vector<vector<int>> upsampleValuesMatrix(height1 * height2, vector<int>(width1 * width2));
    for (int i = 0; i < height1; i++)
    {
        for (int j = 0; j < width1; j++)
        {
            for (int k = 0; k < height2; k++)
            {
                for (int z = 0; z < width2; z++)
                {
                    //cout << i << ", " << j << ", " << k << ", " << z << endl;
                    //cout << derivativeOfAggregation[k][z] << endl;
                    row = i * height2 + k;
                    column = j * width2 + z;
                    upsampleValuesMatrix[row][column] = backPropagatedError[i][j] * derivativeOfAggregation[k][z];
                }
            }
            //cout << backPropagatedError[i][j] << endl;
        }
    }
    return upsampleValuesMatrix;
}

/*
    * Apply the activation function (max) to the features' outuput (index1,index2)
    *
    *
double activation(int index1, int index2, vector<vector<vector<double>>> forwardValues)
{
    double max = 0;
    for (int i = 0; i < forwardValues.size(); i++)
    {
        if (forwardValues[i][index1][index2] > max)
            max = forwardValues[i][index1][index2];
    }
    return max;
}*/

/*
    * backPropagation algorithm implementation, calculates the matrix of values needed for gradient computation
    * @params are the parameters to modify (wieghts or biases)
    */
vector<vector<double>> backPropagation(vector<vector<double>> forwardValues, vector<vector<double>> backwardValues, vector<vector<double>> params)
{
    int count=0;
    vector<vector<double>> deltaErrors(backwardValues.size(), vector<double>(backwardValues[0].size()));
    vector<vector<int>> identity(forwardValues.size(), vector<int>(forwardValues[0].size()));
    for (int i = 0; i < forwardValues.size(); i++)
    {
        for (int j = 0; j < forwardValues[0].size(); j++)
        {
            if (i == j)
                identity[i][j] = 1;
            else
                identity[i][j] = 0;
        }
    }
    //cout << "Identity init" << endl;
    clock_t begin = clock();
    for (int i = 0; i < backwardValues.size(); i++)
    {
        for (int j = 0; j < backwardValues[0].size(); j++)
        {
            count++;
            deltaErrors[i][j] = params[j][i] * backwardValues[i][j] * max((double)0, forwardValues[i][j]) * (identity[i][j] - max((double)0, forwardValues[i][j]));
            cout << '\r' << "Error " << setw(5) << count << " calculated"<< flush;
        }
    }
    cout << endl <<"Time elapsed: " << double(clock() - begin)/CLOCKS_PER_SEC << endl;
    cout << endl;
    return deltaErrors;
}


/*
    * batch gradient descent implementation, calculates the gradients for a layer and updates the weight and biases in the matrices
    *
    */
map<char, vector<vector<double>>> layerUpdater(vector<vector<double>> forwardValues, vector<vector<double>> backwardValues, vector<vector<double>> layerWeights, vector<vector<double>> layerBiases)
{
    cout << "Update started" << endl;
    vector<vector<double>> deltaW(backwardValues.size(), vector<double>(backwardValues[0].size()));
    vector<vector<double>> deltaB(backwardValues.size(), vector<double>(backwardValues[0].size()));
    vector<vector<double>> temp1;
    vector<vector<double>> temp2;
    map<char, vector<vector<double>>> paramMap;
    clock_t begin=clock();
    for (int b = 0; b < batchSize; b++){
        cout << "Calculating error for sample " << b+1 << "/128" << endl;
        cout << "Weights: " << endl;
        temp1=backPropagation(forwardValues, backwardValues, layerWeights);
        cout << "Biases: " << endl;
        temp2=backPropagation(forwardValues, backwardValues, layerWeights);
        for (int i = 0; i < backwardValues.size(); i++){
            for (int j = 0; j < backwardValues[0].size(); j++){
                deltaW[i][j] += temp1[i][j];
                deltaB[i][j] += temp2[i][j];
                //cout << "DeltaW(" << i << ", " << j << "): " << deltaW[i][j] << endl;
                //cout << "DeltaB(" << i << ", " << j << "): " << deltaB[i][j] << endl;
            }
        }
    }
    //for (int b = 0; b < batchSize; b++){
        //cout << "Updating parameters for sample " << b+1 << "/128" << endl;
        for (int i = 0; i < backwardValues.size(); i++){
            for (int j = 0; j < backwardValues[0].size(); j++){
                layerWeights[i][j] = layerWeights[i][j] - (learningRate * (((1 / batchSize) * deltaW[i][j]) + weightDecay * layerWeights[i][j]));
                layerBiases[i][j] = layerBiases[i][j] - (learningRate * ((1 / batchSize) * deltaB[i][j]));
            }
        }
    //}
    paramMap['W'] = layerWeights;
    paramMap['B'] = layerBiases;
    cout << "Update finished" << endl;
    cout << "Total time elapsed: " << double(clock() - begin)/CLOCKS_PER_SEC << endl;
    return paramMap;
}



/*
    * Test
    */
int main(){
    //  NB: IL FOR DEL batchSize DEVE ESSERE FUORI DA backPropagation() IL CHIAMANTE PASSA LE MATRICI DI VALORI E CHIAMA backPropagation() 128 VOLTE CON I DATI GIUSTI
    /*
    vector<vector<vector<double>>> forward{{{1, 2, 3}, {1.3, 1.6, 6.4}, {3.1, 4, 2}},
                                           {{1.3, 1.6, 6.4}, {2.3, 3, 1}, {1, 8.34, 3.456}},
                                           {{1, 6, 2.5}, {3.76, 3, 9.765}, {3, 8, 2.234}}};
    */

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
/*
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
    }*/
}
