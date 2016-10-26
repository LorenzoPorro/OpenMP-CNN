#include <vector>
#include <iostream>

using namespace std;

vector<vector<int>> upsample(vector<vector<int>> &backPropagatedError, vector<vector<int>> &derivativeOfAggregation){
    int height1 = backPropagatedError.size();
    int width1 = backPropagatedError[0].size();
    int height2 = derivativeOfAggregation.size();
    int width2 = derivativeOfAggregation[0].size();
    cout << "h1: " << height1 << endl;
    cout << "h2: " << height2 << endl;
    cout << "w1: " << width1 << endl;
    cout << "w2: " << width2 << endl;
    int row,column;
    vector<vector<int>> upsampleWeightMatrix(height1*height2, vector<int>(width1*width2));
    cout << "vec init" << endl;
    for(int i=0;i<height1;i++){
        for(int j=0;j<width1;j++){
            for(int k=0;k<height2;k++){
                for(int z=0;z<width2;z++){
                    //cout << i << ", " << j << ", " << k << ", " << z << endl;
                    //cout << derivativeOfAggregation[k][z] << endl;
                    row = i*height2+k;
                    column = j*width2+z;
                    upsampleWeightMatrix[row][column]=backPropagatedError[i][j]*derivativeOfAggregation[k][z];
                }
            }
            //cout << backPropagatedError[i][j] << endl;
        }
    }
    return upsampleWeightMatrix;
}

int main(){
    vector<vector<int>> m1 { {1,2}, {3,4} };
    vector<vector<int>> m2 { {1,1}, {1,1} };
    int i,j;
    int count = 0;
    vector<vector<int>> up = upsample(m1,m2);
    for(i=0;i<up.size();i++){
        for(j=0;j<up.size();j++){
            count += 1;
            cout << up[i][j] << " ";
            if(count==up.size()){
                count=0;
                cout << endl;
            }
        }
    }
}
