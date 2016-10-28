#include <vector>
#include <iostream>

public class layerTrainer{

    private:
      int height = 0;
      int width = 0;
      vector<vector<double>> layerWeights;
      vector<vector<double>> layerBiases;
      vecto<vector<vector<double>>> forwardValues;
      vector<vector<double>> backwardValues;
      vector<vector<int>> identity;
      double learningRate;
      double weightDecay;
      int batchSize;

    public:
      layerTrainer(vector<vector<double>> layerWeights, vector<vector<double>> layerBiases, double learningRate, double weightDecay, int batchSize, vector<vector<vector<double>>> forwardValues,vector<vector<double>> backwardValues){
          this.learningRate = learningRate;
          this.weightDecay = weightDecay;
          this.batchSize = batchSize;
          this.layerBiases =layerBiases;
          this.layerWeights = layerWeights;
          this.forwardValues = forwardValues;
          this.backwardValues = backwardValues;
          this.height = layerWeights.size();
          this.width = layerWeights[0].size();
          for(int i=0;i<forwardValues.size();i++){
              for(int j=0;j<forwardValues[0].size();j++){
                  if(i==j)  identity[i][j]=1;
                  else  identity[i][j]=0;
              }
          }
      }

      using namespace std;

      /*
      * upsamples the values to backpropagate to the previous layer, needed only for the pooling layers
      *
      */
      vector<vector<int>> upsample(vector<vector<int>> &backPropagatedError, vector<vector<int>> &derivativeOfAggregation){
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
    void layerUpdater(){
        vector<vector<double>> deltaW(height, vector<double>(width));
        vector<vector<double>> deltaB(height, vector<double>(width));
        for(int b=0;b<batchSize;b++){
            for (int i = 0; i < height; i++){
                for (int j = 0; j < width; j++){
                    deltaW[i][j] += backPropagation();
                    deltaB[i][j] += backPropagation();
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
    }

    /*
    * backPropagation algorithm implementation, calculates the matrix of values needed for gradient computation
    *
    */
    vector<vector<double>> backPropagation(){
        vector<vector<double>> deltaErrors;
        for(int i =0;i<height;i++{
            for(int j=0;j<width;j++){
               deltaErrors[i][j]=layerWeights[j][i]*backwardValues[i][j]*activation(i,j)*(identity[i][j]-activation(i,j));   
            }
        }
        return deltaErrors;
    }

    /*
    * Apply the activation function (max) to the features' outuput (index1,index2)
    *
    */
    double activation(int index1, int index2){
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
}
