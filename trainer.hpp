

class Trainer{
    private:
      int height;
      int width;
      vector<vector<double>> layerWeights;
      vector<vector<double>> layerBiases;
      vecto<vector<vector<double>>> forwardValues;
      vector<vector<double>> backwardValues;
      vector<vector<int>> identity;
      double learningRate;
      double weightDecay;
      int batchSize;
    public:
      Trainer(vector<vector<double>> layerWeights, vector<vector<double>> layerBiases, double learningRate, double weightDecay, int batchSize, vector<vector<vector<double>>> forwardValues,vector<vector<double>> backwardValues);
      
      vector<vector<int>> upsample(vector<vector<int>> &backPropagatedError, vector<vector<int>> &derivativeOfAggregation);

      
};