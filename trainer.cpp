#include <vector>
#include <iostream>
#include <map>

using namespace std;

class layerTrainer{

    private:
      int height = 0;
      int width = 0;
      double learningRate;
      double weightDecay;
      int batchSize;

    public:
      Trainer::Trainer(double learningRate, double weightDecay, int batchSize){
          learningRate <- learningRate;
          weightDecay <- weightDecay;
          batchSize <- batchSize;
      }

      using namespace std;

      /*
      * upsamples the values to backpropagate to the previous layer, needed only for the pooling layers
      *
      */
      vector<vector<int>> Trainer::upsample(vector<vector<int>> &backPropagatedError, vector<vector<int>> &derivativeOfAggregation){
          int height1 = backPropagatedError.size();
          int width1 = backPropagatedError[0].size();
          int height2 = derivativeOfAggregation.size();
          int width2 = derivativeOfAggregation[0].size();
          int row, column;
          vector<vector<int>> upsampleValuesMatrix(height1 * height2, vector<int>(width1 * width2));
          for (int i = 0; i < height1; i++){
              for (int j = 0; j < width1; j++){
                  for (int k = 0; k < height2; k++){
                      for (int z = 0; z < width2; z++){
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
    * batch gradient descent implementation, calculates the gradients for a layer and updates the weight and biases in the matrices
    *
    */
     map<char, vector<vector<double>>> Trainer::layerUpdater(vecto<vector<vector<double>>> forwardValues, vector<vector<double>> backwardValues, vector<vector<double>> &layerWeights, vector<vector<double>> &layerBiases){
        vector<vector<double>> deltaW(height, vector<double>(width));
        vector<vector<double>> deltaB(height, vector<double>(width));
        map<char, vector<vector<double>>> paramMap;
        for(int b=0;b<batchSize;b++){
            for (int i = 0; i < height; i++){
                for (int j = 0; j < width; j++){
                    deltaW[i][j] += backPropagation(forwardValues, backwardValues, layerWeights);
                    deltaB[i][j] += backPropagation(forwardValues, backwardValues, layerWeights);
                }
            }
        }
        for(int b=0;b<batchSize;b++){
            for(int i=0;i<height;i++){
                for(int j=0;j<width;j++){
                    layerWeights[i][j]=layerWeights[i][j]-(learningRate*(((1/batchSize)*deltaW)+weightDecay*layerWeights[i][j]));
                    layerBiases[i][j]=layerBiases[i][j]-(learningRate*((1/batchSize)*deltaB));
                }
            }
        }
        paramMap['W']=layerWeights;
        paramMap['B']=layerBiases;
        return paramMap;
    }

    /*
    * backPropagation algorithm implementation, calculates the matrix of values needed for gradient computation
    *
    */
    vector<vector<double>> backPropagation(vecto<vector<vector<double>>> forwardValues, vector<vector<double>> backwardValues, vector<vector<double>> &layerWeights){
        vector<vector<double>> deltaErrors;
        vector<vector<int>> identity(forwardValues.size(), vector<double>(forwardValues[0].size()));
        for(int i=0;i<forwardValues.size();i++){
              for(int j=0;j<forwardValues[0].size();j++){
                  if(i==j)  identity[i][j]=1;
                  else  identity[i][j]=0;
              }
        }
        for(int i =0;i<height;i++){
            for(int j=0;j<width;j++){
               deltaErrors[i][j]=layerWeights[j][i]*backwardValues[i][j]*activation(i,j, forwardValues)*(identity[i][j]-activation(i,j, forwardValues));   
            }
        }
        return deltaErrors;
    }

    /*
    * Apply the activation function (max) to the features' outuput (index1,index2)
    *
    */
    double activation(int index1, int index2, vector<vector<vector<double>>> forwardValues){
        double max=0;
        for(int i=0;i<forwardValues.size();i++){
            if(forwardValues[i][index1][index2]>max)  max=forwardValues[i][index1][index2];
        }
        return max;
    }
    
    /*
    * Test
    */
    int main(){
        Trainer train(0.01,0.0005,128);
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
