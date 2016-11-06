
#include <vector>
#include <map>

using namespace std;

#ifndef INPUT_READER_HPP
#define INPUT_READER_HPP

vector<int> split(string s, char c);

map<char, vector<vector <int >>> getColors(string filename);

vector<map<char,vector<vector<int>>>> getInputArray();

#endif